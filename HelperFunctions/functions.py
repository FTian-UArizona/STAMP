# Library

import torch
import math
import pandas as pd
from torch.distributions import Laplace
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import vonmises_fisher
from typing import List, Callable, Optional, Tuple, Any, Dict

import torch.nn.functional as F

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

# This is the function for vmf mechanism

def add_noise_polar(
    token_ids,
    eps_mag_per_token,
    eps_dir_per_token,
    embedding_table,
    tokenizer,
):
    decoded_sentences = []
    special_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}

    # Sensitivity for noise scale
    r_norms = torch.norm(embedding_table, dim=-1)
    delta_mag = (r_norms.max() - r_norms.min()).item()
    delta_dir = math.pi

    # Normalize embedding table for search
    norm_embedding_table = torch.nn.functional.normalize(embedding_table, dim=1)

    # Handle single/batched inputs
    if isinstance(token_ids, torch.Tensor):
        token_ids = [token_ids] if token_ids.dim() == 1 else list(token_ids)

    for tokens in token_ids:
        laplace_scale = delta_mag / eps_mag_per_token
        kappa = eps_dir_per_token / delta_dir
        emb = embedding_table[tokens]
        noisy_embs = []

        for e in emb:
            # 1) Polar decomposition
            r = torch.norm(e)
            u = e / (r + 1e-8) # cannot be zero

            # 2) Magnitude noise (Laplace)
            mag_noise = Laplace(0.0, laplace_scale).sample()
            new_r = torch.clamp(r + mag_noise, min=1e-6, max=1.0)
            # Set magnitude to 1
            # new_r = 1

            # 3) Directional noise (vMF)
            u_np = u.detach().cpu().numpy()
            new_u_np = vonmises_fisher.rvs(mu=u_np, kappa=kappa)

            # fix shape mismatch
            if new_u_np.ndim > 1:
                new_u_np = new_u_np.squeeze(0)
            new_u = torch.tensor(new_u_np, device=u.device, dtype=u.dtype)

            # 4) Reconstruct noisy embedding
            noisy_vec = new_r * new_u
            noisy_vec = noisy_vec / (torch.norm(noisy_vec) + 1e-8)
            noisy_embs.append(noisy_vec)

        noisy_embs = torch.stack(noisy_embs, dim=0)

        # 5) Closest point search (Euclidean on normalized embeddings)
        diff = noisy_embs.unsqueeze(1) - norm_embedding_table.unsqueeze(0)
        distances = torch.norm(diff, dim=2)

        # Mask special tokens
        special_ids_to_mask = [i for i in special_ids if i < distances.shape[1]]
        if special_ids_to_mask:
            distances[:, special_ids_to_mask] = float("inf")

        new_tokens = torch.argmin(distances, dim=1)

        # Decode
        orig_sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        noisy_sentence = tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded_sentences.append(noisy_sentence)

    return decoded_sentences


# This is the function for vmf mechanism with fixed magnitude

def add_noise_polar_fixed_r(
    token_ids,
    eps_dir_per_token,
    embedding_table,
    tokenizer,
    special_ids=safe_special_ids   
):
    decoded_sentences = []
    special_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}

    # Sensitivity for noise scale
    r_norms = torch.norm(embedding_table, dim=-1)
    delta_mag = (r_norms.max() - r_norms.min()).item()
    delta_dir = math.pi

    # Normalize embedding table for search
    norm_embedding_table = torch.nn.functional.normalize(embedding_table, dim=1)

    # Handle single/batched inputs
    if isinstance(token_ids, torch.Tensor):
        token_ids = [token_ids] if token_ids.dim() == 1 else list(token_ids)

    for tokens in token_ids:
        kappa = eps_dir_per_token / delta_dir
        emb = embedding_table[tokens]
        noisy_embs = []

        for e in emb:
            # 1) Polar decomposition
            r = torch.norm(e)
            u = e / (r + 1e-8) # cannot be zero

            # 2) Magnitude noise (Laplace)
            # Set magnitude to 1
            new_r = 1

            # 3) Directional noise (vMF)
            u_np = u.detach().cpu().numpy()
            new_u_np = vonmises_fisher.rvs(mu=u_np, kappa=kappa)

            # fix shape mismatch
            if new_u_np.ndim > 1:
                new_u_np = new_u_np.squeeze(0)
            new_u = torch.tensor(new_u_np, device=u.device, dtype=u.dtype)

            # 4) Reconstruct noisy embedding
            noisy_vec = new_r * new_u
            noisy_vec = noisy_vec / (torch.norm(noisy_vec) + 1e-8)
            noisy_embs.append(noisy_vec)
            # Clear cache to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        noisy_embs = torch.stack(noisy_embs, dim=0)

        # 5) Closest point search (Euclidean on normalized embeddings)
        diff = noisy_embs.unsqueeze(1) - norm_embedding_table.unsqueeze(0)
        distances = torch.norm(diff, dim=2)

        # Mask special tokens
        special_ids_to_mask = [i for i in special_ids if i < distances.shape[1]]
        if special_ids_to_mask:
            distances[:, special_ids_to_mask] = float("inf")

        new_tokens = torch.argmin(distances, dim=1)

        # Clear cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Decode
        orig_sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        noisy_sentence = tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded_sentences.append(noisy_sentence)

        # Clear cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return decoded_sentences


# This is the function for Laplace mechanism

def add_noise_isotropic_laplace(
    token_ids,
    eps_per_token,
    embedding_table,
    tokenizer,
    normalize_before_decode=False,
    delta1_L1: float = None,   # optional: pass your own L1 sensitivity if you prefer clipping
):
    device = embedding_table.device
    decoded_sentences = []
    special_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}

    # --- Sensitivity (Δ1) for L1 metric ---
    if delta1_L1 is None:
        # Upper bound: sum of per-dimension ranges across the vocab table
        col_max = embedding_table.max(dim=0).values
        col_min = embedding_table.min(dim=0).values
        delta1_L1 = torch.sum(col_max - col_min).item()

    # --- Laplace scale per coordinate ---
    b = float(delta1_L1) / float(eps_per_token)
    lap = Laplace(loc=torch.tensor(0.0, device=device), scale=torch.tensor(b, device=device))

    # --- Normalize vocab if requested (since we are doing polar decode) ---
    if normalize_before_decode:
        vocab_for_search = F.normalize(embedding_table, p=2, dim=1)
    else:
        vocab_for_search = embedding_table

    # Handle single/batched inputs
    if isinstance(token_ids, torch.Tensor):
        token_ids = [token_ids] if token_ids.dim() == 1 else list(token_ids)

    for tokens in token_ids:
        # 1) Gather embeddings for this sequence  [L, d]
        emb = embedding_table[tokens.to(device)]

        # 2) Add isotropic Laplace noise  [L, d]
        noise = lap.sample(sample_shape=emb.shape).to(emb.dtype)
        noisy_embs = emb + noise

        # 3) Optional normalization before NN search
        if normalize_before_decode:
            noisy_embs = F.normalize(noisy_embs, p=2, dim=1)

        # 4) Nearest-neighbor decode (Euclidean on whatever space we chose)
        #    distances: [L, V]
        diff = noisy_embs.unsqueeze(1) - vocab_for_search.unsqueeze(0)
        distances = torch.norm(diff, dim=2)

        # Mask special tokens (avoid mapping into pad/eos)
        special_ids_to_mask = [i for i in special_ids if (i is not None) and (i < distances.shape[1])]
        if special_ids_to_mask:
            distances[:, special_ids_to_mask] = float("inf")

        new_tokens = torch.argmin(distances, dim=1)

        # 5) Decode back to text
        noisy_sentence = tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded_sentences.append(noisy_sentence)

    return decoded_sentences


# This is the function for Mask and fill functions
def epsilon_zero_mask(
    tokens: List[str],
    mask_positions: List[int],
    mask_token: str = "[MASK]"
) -> List[str]:

    if not mask_positions:
        return list(tokens)
    mask_set = set(mask_positions)
    return [mask_token if i in mask_set else tok for i, tok in enumerate(tokens)]

# masked_tokens = epsilon_zero_mask(tokens, mask_positions)

def epsilon_zero_mask_and_fill(
    tokens: List[str],
    mask_positions: List[int],
    filler: Optional[Callable[[List[str], List[int]], List[str]]] = None,
    mask_token: str = "[MASK]"
) -> List[str]:
    """
    ε=0 mask-and-fill:
      1) mask tokens at `mask_positions` (perfect privacy for originals),
      2) optionally fill those masks using `filler(tokens_with_masks, mask_positions)`.
    NOTE: The filler MUST NOT access the original tokens to keep ε=0 intact.
    """
    masked = epsilon_zero_mask(tokens, mask_positions, mask_token=mask_token)

    if filler is None or not mask_positions:
        return masked

    fills = filler(masked, mask_positions)
    if len(fills) != len(mask_positions):
        raise ValueError("filler must return exactly one string per masked position.")

    out = list(masked)
    for rep, idx in zip(fills, mask_positions):
        out[idx] = rep
    return out

# filled_tokens = epsilon_zero_mask_and_fill(tokens, mask_positions, filler=my_filler_fn)


# Apply STAMP here

def apply_stamp(
    token_ids,
    group_assignments,
    embedding_table,
    tokenizer,
    eps_dir_by_group: dict,      # REQUIRED: {1:ε1, 2:ε2, 3:ε3, 4:ε4}
):
    """
    STAMP (strict): every token is privatized with the ε for its group.
    Backward-compatible signature, but only `eps_dir_by_group` is used.
    Returns a list of decoded sentences (batch-aligned).
    """
    import torch

    # normalize batch
    if isinstance(token_ids, torch.Tensor):
        batch = [token_ids] if token_ids.dim() == 1 else list(token_ids)
    elif isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], int):
        batch = [token_ids]
    else:
        batch = token_ids

    decoded_sentences = []

    for tokens in batch:
        toks = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        if len(group_assignments) != len(toks):
            raise ValueError("group_assignments must have one entry per token.")

        out_ids = []
        for i, tok in enumerate(toks):
            g = group_assignments[i]
            eps_g = float(eps_dir_by_group[g])  # must exist per group
            noisy_text = add_noise_polar_fixed_r(
                torch.tensor([tok]),
                eps_g,
                embedding_table,
                tokenizer,
            )[0]  # returns a single decoded string
            out_ids.extend(tokenizer.encode(noisy_text, add_special_tokens=False))

        decoded_sentences.append(
            tokenizer.decode(out_ids, skip_special_tokens=True)
        )

    return decoded_sentences

# This is the laplace version 
def apply_stamp_laplace(
    token_ids,
    group_assignments,
    embedding_table,
    tokenizer,
    eps_dir_by_group: dict,      # REQUIRED: {1:ε1, 2:ε2, 3:ε3, 4:ε4}
):
    """
    STAMP (strict): every token is privatized with the ε for its group.
    Backward-compatible signature, but only `eps_dir_by_group` is used.
    Returns a list of decoded sentences (batch-aligned).
    """
    import torch

    # normalize batch
    if isinstance(token_ids, torch.Tensor):
        batch = [token_ids] if token_ids.dim() == 1 else list(token_ids)
    elif isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], int):
        batch = [token_ids]
    else:
        batch = token_ids

    decoded_sentences = []

    for tokens in batch:
        toks = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        if len(group_assignments) != len(toks):
            raise ValueError("group_assignments must have one entry per token.")

        out_ids = []
        for i, tok in enumerate(toks):
            g = group_assignments[i]
            eps_g = float(eps_dir_by_group[g])  # must exist per group
            noisy_text = add_noise_isotropic_laplace(
                torch.tensor([tok]),
                eps_g,
                embedding_table,
                tokenizer,
            )[0]  # returns a single decoded string
            out_ids.extend(tokenizer.encode(noisy_text, add_special_tokens=False))

        decoded_sentences.append(
            tokenizer.decode(out_ids, skip_special_tokens=True)
        )

    return decoded_sentences



# Simpler cases
# So how about a magic trick:

def apply_stamp_allow_inf_laplace(
    token_ids,
    group_assignments,
    embedding_table,
    tokenizer,
    *,
    eps_dir_by_group: dict,         # e.g., {1:200, 2:float("inf"), 3:400, 4:600}
    special_ids=None,               # optional override; defaults to tokenizer.all_special_ids
):
    """
    STAMP with ∞-budget pass-through:
      - If eps(g) is +∞ -> return the exact original token (no change)
      - Else            -> apply add_noise_polar_fixed_r with that ε

    Returns: [decoded_string] (batch-aligned), same shape as your other apply_*.
    """
    # sanitize specials (avoid None)
    if special_ids is None:
        special_ids = [sid for sid in (getattr(tokenizer, "all_special_ids", []) or []) if isinstance(sid, int)]
    else:
        special_ids = [sid for sid in (special_ids or []) if isinstance(sid, int)]

    # normalize batch
    if isinstance(token_ids, torch.Tensor):
        batch = [token_ids] if token_ids.dim() == 1 else list(token_ids)
    elif isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], int):
        batch = [token_ids]
    else:
        batch = token_ids

    decoded = []

    for seq in batch:
        ids = seq.tolist() if hasattr(seq, "tolist") else list(seq)
        if len(ids) != len(group_assignments):
            raise ValueError("group_assignments must align 1:1 with token_ids")

        out_ids = []
        for i, tok in enumerate(ids):
            g = group_assignments[i]
            eps_g = eps_dir_by_group[g]

            # Pass-through if ε is +∞ (or explicitly math.inf as float)
            if isinstance(eps_g, (float, int)) and math.isinf(float(eps_g)) and float(eps_g) > 0:
                out_ids.append(tok)  # exact token id unchanged
                continue

            # Otherwise: apply Laplace noise
            noisy_text = add_noise_isotropic_laplace(
                torch.tensor([tok]),
                eps_g,
                embedding_table,
                tokenizer,
            )[0] 
            out_ids.extend(tokenizer.encode(noisy_text, add_special_tokens=False))

        decoded.append(tokenizer.decode(out_ids, skip_special_tokens=True))

    return decoded



def apply_stamp_allow_inf_polar(
    token_ids,
    group_assignments,
    embedding_table,
    tokenizer,
    *,
    eps_dir_by_group: dict,         # e.g., {1:200, 2:float("inf"), 3:400, 4:600}
    special_ids=None,               # optional override; defaults to tokenizer.all_special_ids
):
    """
    STAMP with ∞-budget pass-through:
      - If eps(g) is +∞ -> return the exact original token (no change)
      - Else            -> apply add_noise_polar_fixed_r with that ε

    Returns: [decoded_string] (batch-aligned), same shape as your other apply_*.
    """
    # sanitize specials (avoid None)
    if special_ids is None:
        special_ids = [sid for sid in (getattr(tokenizer, "all_special_ids", []) or []) if isinstance(sid, int)]
    else:
        special_ids = [sid for sid in (special_ids or []) if isinstance(sid, int)]

    # normalize batch
    if isinstance(token_ids, torch.Tensor):
        batch = [token_ids] if token_ids.dim() == 1 else list(token_ids)
    elif isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], int):
        batch = [token_ids]
    else:
        batch = token_ids

    decoded = []

    for seq in batch:
        ids = seq.tolist() if hasattr(seq, "tolist") else list(seq)
        if len(ids) != len(group_assignments):
            raise ValueError("group_assignments must align 1:1 with token_ids")

        out_ids = []
        for i, tok in enumerate(ids):
            g = group_assignments[i]
            eps_g = eps_dir_by_group[g]

            # Pass-through if ε is +∞ (or explicitly math.inf as float)
            if isinstance(eps_g, (float, int)) and math.isinf(float(eps_g)) and float(eps_g) > 0:
                out_ids.append(tok)  # exact token id unchanged
                continue

            # Otherwise: apply directional noise (vMF; κ = ε/π inside)
            noisy_text = add_noise_polar_fixed_r(
                torch.tensor([tok], device=embedding_table.device),
                float(eps_g),
                embedding_table,
                tokenizer,
                special_ids=special_ids,
            )[0]
            out_ids.extend(tokenizer.encode(noisy_text, add_special_tokens=False))

        decoded.append(tokenizer.decode(out_ids, skip_special_tokens=True))

    return decoded

# Drop tokens

def apply_drop_zero_keep_inf(
    token_ids,
    group_assignments: List[int],
    tokenizer,
    *,
    eps_dir_by_group: Dict[int, float],
):
    """
    Keep tokens iff group's ε is +∞.
    Drop tokens iff group's ε == 0.
    Leave other finite non-zero ε tokens unchanged (pass-through).

    Returns a list of decoded strings, one per input sequence (batch-friendly).
    """
    # normalize batch
    if isinstance(token_ids, torch.Tensor):
        batch = [token_ids] if token_ids.dim() == 1 else list(token_ids)
    elif isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], int):
        batch = [token_ids]
    else:
        batch = token_ids

    decoded = []
    for seq in batch:
        ids = seq.tolist() if hasattr(seq, "tolist") else list(seq)
        if len(ids) != len(group_assignments):
            raise ValueError("group_assignments must align 1:1 with token_ids")

        out_ids = []
        for i, tok in enumerate(ids):
            g = group_assignments[i]
            eps_g = eps_dir_by_group[g]
            # Drop if ε == 0
            if isinstance(eps_g, (int, float)) and float(eps_g) == 0.0:
                continue
            # Keep if ε = +∞
            if isinstance(eps_g, (int, float)) and math.isinf(float(eps_g)) and float(eps_g) > 0:
                out_ids.append(tok)
                continue
            # Otherwise (finite, non-zero): pass through unchanged
            out_ids.append(tok)

        decoded.append(tokenizer.decode(out_ids, skip_special_tokens=True))

    return decoded
# Pertube each token (PREVIOUS versions, leave them here)

def unit_test_token_changes(context: str, eps_dir: float = 500):
    """Perturb each token directionally and show before/after tokens."""
    # 1) tokenize (no special tokens)
    ids = tokenizer.encode(context, add_special_tokens=False)
    toks = tokenizer.convert_ids_to_tokens(ids)

    # 2) eps per token (same epsilon for all to keep it simple)
    eps_per_token = eps_dir

    # 3) safe special ids (avoid the None issue)
    special_ids = [sid for sid in (getattr(tokenizer, "all_special_ids", []) or []) if isinstance(sid, int)]

    # 4) run your polar angular perturbation on each token individually
    new_tokens = []
    for tid in ids:
        one_tok = torch.tensor([tid], dtype=torch.long)
        # Call mechanism on JUST THIS TOKEN with that token's ε
        noisy_piece = functions.add_noise_polar_fixed_r([one_tok], eps_per_token, embedding_table, tokenizer, special_ids=special_ids)[0]  # string for that token
        new_tokens.append(noisy_piece.strip())

    # 5) print a compact diff
    print(f"Context (first 200 chars): {context[:200]} ...\n")
    print(f"{'idx':>3}  {'orig':<18} -> {'perturbed':<18}")
    print("-"*60)
    for i, tok in enumerate(toks):
        # new_tokens[i] is the decoded token replacement for position i
        new_tok = new_tokens[i]
        print(f"{i:>3}  {tok:<18} -> {new_tok:<18}")