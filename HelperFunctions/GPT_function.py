# Library

import torch
import math
import re
import numpy as np
import pandas as pd
from torch.distributions import Laplace

from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import vonmises_fisher
import torch.nn.functional as F

from typing import List, Callable, Optional, Tuple, Any, Dict

from collections import Counter

# Remove this atfer
from openai import OpenAI
client = OpenAI(api_key="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")  # needs OPENAI_API_KEY


# --- Load tokenizer and GPT-2 model ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# --- Extract embedding table ---
embedding_table = model.transformer.wte.weight.detach()
# Normalize embedding table for search
norm_embedding_table = torch.nn.functional.normalize(embedding_table, dim=1)

# Why is this?
safe_special_ids = [i for i in tokenizer.all_special_ids if isinstance(i, int)]

# --- Check neighbors for a few words ---
def find_neighbors(word, k=15):
    """
    Finds top-k neighbors for a word using cosine similarity on normalized embeddings.
    """
    token_id = tokenizer.encode(word, add_special_tokens=False)[0]
    norm_table = torch.nn.functional.normalize(embedding_table, dim=1)
    vec = norm_table[token_id].unsqueeze(0)
    sims = torch.matmul(vec, norm_table.T).squeeze(0)

    top_k = torch.topk(sims, k + 1)  # include self
    neighbors = []
    for idx, sim in zip(top_k.indices.tolist(), top_k.values.tolist()):
        if idx != token_id:
            neighbors.append((tokenizer.decode([idx])))
            # , round(float(sim), 4)))
        if len(neighbors) == k:
            break
    return neighbors


# Musk and fill

@torch.no_grad()
def apply_drop_zero_keep_inf_gpt2fill_ids(
    token_ids: list[int] | torch.Tensor,
    groups: list[int],
    eps_dir_by_group: dict[int, float],
    *,
    tokenizer,           # base tokenizer (used by your model/embeddings)
    gpt2_tok,            # GPT-2 tokenizer
    gpt2_model,          # GPT-2 (causal LM) model
    deterministic: bool = False,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 50,
    context_window: int = 256,      # left tokens for GPT-2 prompt (after re-tokenizing with GPT-2)
) -> tuple[str, list[int]]:
    """
    ε == 0  -> drop token and insert ONE GPT-2 token predicted from left context only
    ε = +∞  -> keep token unchanged
    else    -> pass through unchanged

    Returns: (new_text, new_ids) in the BASE tokenizer space.
    """
    # normalize input
    if isinstance(token_ids, torch.Tensor):
        base_ids: list[int] = token_ids.tolist()
    else:
        base_ids = list(token_ids)

    assert len(base_ids) == len(groups), "groups must align 1:1 with token_ids"

    device = next(gpt2_model.parameters()).device
    # Ensure GPT-2 has a pad token for generate()
    if getattr(gpt2_tok, "pad_token_id", None) is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token

    out_ids: list[int] = []

    for i, tok in enumerate(base_ids):
        g = groups[i]
        eps_g = eps_dir_by_group[g]

        # Case A: ε == 0  → drop & fill one GPT-2 token
        if isinstance(eps_g, (int, float)) and float(eps_g) == 0.0:
            # 1) Build left context as TEXT via base tokenizer
            left_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            # 2) Retokenize left context with GPT-2, taking only the most recent context_window tokens
            gpt2_prompt_ids = gpt2_tok(left_text, return_tensors="pt").to(device)
            if context_window is not None and gpt2_prompt_ids.input_ids.shape[1] > context_window:
                gpt2_prompt_ids = {
                    "input_ids": gpt2_prompt_ids.input_ids[:, -context_window:],
                    "attention_mask": gpt2_prompt_ids.attention_mask[:, -context_window:],
                }

            # Seed if empty
            if gpt2_prompt_ids["input_ids"].numel() == 0:
                seed_id = gpt2_tok.eos_token_id if gpt2_tok.eos_token_id is not None else gpt2_tok.bos_token_id
                if seed_id is None:
                    seed_id = gpt2_tok.encode(" ", add_special_tokens=False)[0]
                gpt2_prompt_ids = {
                    "input_ids": torch.tensor([[seed_id]], device=device),
                    "attention_mask": torch.tensor([[1]], device=device),
                }

            # 3) Generate exactly ONE GPT-2 token
            if deterministic:
                logits = gpt2_model(input_ids=gpt2_prompt_ids["input_ids"]).logits[:, -1, :]
                # discourage EOS as a fill
                eos_id = gpt2_tok.eos_token_id
                if eos_id is not None:
                    logits[:, eos_id] = -float("inf")
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                gen = gpt2_model.generate(
                    **gpt2_prompt_ids,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=gpt2_tok.pad_token_id,
                    eos_token_id=gpt2_tok.eos_token_id,
                    bad_words_ids=[[gpt2_tok.eos_token_id]] if gpt2_tok.eos_token_id is not None else None,
                )
                next_id = int(gen[0, -1].item())

            # 4) Decode that GPT-2 token to TEXT, then re-encode with BASE tokenizer
            piece_text = gpt2_tok.decode([next_id], skip_special_tokens=True).strip()
            if not piece_text:
                piece_text = " the"  # safe, neutral fallback
            rep_ids = tokenizer.encode(piece_text, add_special_tokens=False)
            if not rep_ids:  # in rare cases decode→re-encode yields no ids
                rep_ids = tokenizer.encode(" the", add_special_tokens=False)
            out_ids.extend(rep_ids)
            continue

        # Case B: ε = +∞ → keep token unchanged
        if isinstance(eps_g, (int, float)) and math.isinf(float(eps_g)) and float(eps_g) > 0:
            out_ids.append(tok)
            continue

        # Case C: finite non-zero → pass through unchanged
        out_ids.append(tok)

    new_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return new_text, out_ids



@torch.no_grad()
def apply_drop_zero_keep_inf_gpt2fill(
    context_text: str,
    groups: list[int],
    eps_dir_by_group: dict[int, float],
    *,
    tokenizer,     # base tokenizer
    gpt2_tok,      # GPT-2 tokenizer
    gpt2_model,    # GPT-2 model
    **gen_kwargs,
) -> str:
    """
    Tokenizes with BASE tokenizer, calls *_ids, returns TEXT in BASE space.
    `groups` must match the BASE tokenization of `context_text`.
    """
    base_ids = tokenizer.encode(context_text, add_special_tokens=False)
    assert len(base_ids) == len(groups), "groups length must match base tokenization length"
    new_text, _ = apply_drop_zero_keep_inf_gpt2fill_ids(
        base_ids, groups, eps_dir_by_group,
        tokenizer=tokenizer, gpt2_tok=gpt2_tok, gpt2_model=gpt2_model,
        **gen_kwargs
    )
    return new_text


# GPT-4 drop+fill baseline: drop ε=0 (fill from LEFT context only), keep ε=+∞, pass-through otherwise
# Returns (new_text, new_ids) in the BASE tokenizer space.

def _sanitize_special_ids(tokenizer, ids: List[int]) -> List[int]:
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    return [i for i in ids if i not in special]

def _decode_base(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(_sanitize_special_ids(tokenizer, ids), skip_special_tokens=True)

@torch.no_grad()
def apply_drop_zero_keep_inf_gpt4fill_ids(
    token_ids: List[int] | torch.Tensor,
    groups: List[int],
    eps_dir_by_group: Dict[int, float],
    *,
    tokenizer,                  # base tokenizer (same one used by your embeddings)
    openai_client,              # OpenAI() client (from openai import OpenAI; client=OpenAI(...))
    model: str = "gpt-4o-mini", # or "gpt-4o"
    temperature: float = 0.2,
    top_p: float = 0.9,
    context_window: int = 256,  # number of BASE tokens to include from the LEFT (already-built out_ids)
    enforce_one_token: bool = True,  # keep exactly 1 base token from the GPT-4 output
    mask_groups: Optional[set] = None,  # optionally force-mask these groups irrespective of eps map
) -> Tuple[str, List[int]]:
    """
    ε == 0  -> drop token and insert ONE GPT-4 suggestion (text → re-encode into BASE tokenizer)
    ε = +∞  -> keep token unchanged
    else    -> pass-through unchanged

    LEFT-context-only, to match your GPT-2 baseline invariants.
    """
    # normalize input ids
    if isinstance(token_ids, torch.Tensor):
        base_ids: List[int] = token_ids.tolist()
    else:
        base_ids = list(token_ids)

    assert len(base_ids) == len(groups), "groups must align 1:1 with token_ids"
    if mask_groups is None:
        mask_groups = set()

    out_ids: List[int] = []

    for i, tok in enumerate(base_ids):
        g = groups[i]
        eps_g = eps_dir_by_group.get(g, float("inf"))
        if g in mask_groups:
            eps_g = 0.0

        # Case A: ε == 0  → drop & GPT-4 fill (from LEFT context only)
        if isinstance(eps_g, (int, float)) and float(eps_g) == 0.0:
            # Build LEFT text from already-output BASE tokens (keeps alignment stable)
            left_ids  = out_ids[-context_window:] if context_window else out_ids
            left_text = _decode_base(tokenizer, left_ids)

            # Ask GPT-4 for a minimal fill
            sys = ("You fill a single masked position using ONLY the left context. "
                   "Return ONLY the replacement word or short phrase; no quotes, no punctuation, no extra text.")
            user = (f"Left context:\n{left_text}\n\n"
                    f"Output ONLY the replacement {'word' if enforce_one_token else 'short phrase (<=3 words)'} "
                    "for the next token position.")

            resp = openai_client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":sys},
                          {"role":"user","content":user}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=8,
            )
            piece_text = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
            if not piece_text:
                piece_text = " the"  # conservative fallback

            # Re-encode with BASE tokenizer
            rep_ids = tokenizer.encode(piece_text, add_special_tokens=False)
            rep_ids = _sanitize_special_ids(tokenizer, rep_ids)
            if not rep_ids:
                rep_ids = tokenizer.encode(" the", add_special_tokens=False)

            if enforce_one_token:
                rep_ids = [rep_ids[0]]

            out_ids.extend(rep_ids)
            continue

        # Case B: ε = +∞ → keep token unchanged
        if isinstance(eps_g, (int, float)) and math.isinf(float(eps_g)) and float(eps_g) > 0:
            out_ids.append(tok)
            continue

        # Case C: finite non-zero → pass-through unchanged (plug STAMP noise here if desired)
        out_ids.append(tok)

    new_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return new_text, out_ids



# convenience wrapper that takes TEXT and returns TEXT (BASE space)
@torch.no_grad()
def apply_drop_zero_keep_inf_gpt4fill(
    context_text: str,
    groups: List[int],
    eps_dir_by_group: Dict[int, float],
    *,
    tokenizer,
    openai_client,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    top_p: float = 0.9,
    context_window: int = 256,
    enforce_one_token: bool = True,
    mask_groups: Optional[set] = None,
) -> str:
    base_ids = tokenizer.encode(context_text, add_special_tokens=False)
    assert len(base_ids) == len(groups), "groups length must match base tokenization length"
    new_text, _ = apply_drop_zero_keep_inf_gpt4fill_ids(
        base_ids, groups, eps_dir_by_group,
        tokenizer=tokenizer,
        openai_client=openai_client,
        model=model,
        temperature=temperature,
        top_p=top_p,
        context_window=context_window,
        enforce_one_token=enforce_one_token,
        mask_groups=mask_groups,
    )
    return new_text

