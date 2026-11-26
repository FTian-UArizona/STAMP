# STAMP: Selective Task-Aware Mechanism for Text Privacy

This repository contains the **reference implementation** for the paper

> **STAMP: Selective Task-Aware Mechanism for Text Privacy**  
> Fengwei Tian, Payel Bhattacharjee, Heidi Hanson, Geoffrey D. Rubin, Joseph Y. Lo, Ravi Tandon  
> EACL 2026 (Long Papers), **Oral** · [Paper](https://aclanthology.org/2026.eacl-long.61/)

STAMP is a framework for **task-aware, sensitivity-aware privacy budgeting** for text under local differential privacy (LDP). It combines:

* A **2×2 grouping** of tokens by *sensitivity* (e.g., PII) and *task importance*
* A **Polar mechanism** that adds noise only to the **direction** of token embeddings (on the unit sphere)
* Standard **metric-LDP** guarantees per token, lifted to sequences via parallel composition

We compare STAMP against a **uniform budget** (the same ε for every token), and the Polar mechanism against an **isotropic Laplace** mechanism.

---

## Repository structure

The most relevant files are:

* `functions.py`
  Core library for STAMP and baselines:

  * Polar mechanism (vMF-based, direction-only noise)
  * Isotropic Laplace mechanism on embeddings
  * STAMP wrappers (2×2 grouping + per-group budgets)
  * Utility helpers (nearest-neighbor decoding, optional ε=0/∞ variants)

* `GPT_function.py`
  Optional **drop+fill baselines** (not required for main results):

  * GPT-2 “drop sensitive token and fill from left context”
  * GPT-4 “drop sensitive token and fill from left context”
    These require a local GPT-2 model / an OpenAI API key respectively.

* `fantasy_squad.json`
  Synthetic SQuAD-style QA dataset (same JSON structure as SQuAD: `data -> paragraphs -> {context, qas}`) used for the QA experiments. See [SQuAD vs. Fantasy SQuAD](#squad-vs-fantasy-squad) below.

* Notebooks (for experiments and figures)

---

## Datasets

The notebooks use standard Hugging Face datasets plus one synthetic QA set:

* **AG News**: `datasets.load_dataset("ag_news")`
* **Yelp**: `datasets.load_dataset("Yelp/yelp_review_full")`
* **Fantasy SQuAD**: `Fantasy_Dataset/fantasy_squad.json` in this repo, also on Hugging Face as
  [`Setpember/Fantasy-SQUAD_10`](https://huggingface.co/datasets/Setpember/Fantasy-SQUAD_10)

### SQuAD vs. Fantasy SQuAD

The paper reports question answering (QA) on SQuAD (Rajpurkar et al., 2016). SQuAD is built from
Wikipedia, and the LLM that answers the questions (gpt-4o-mini) has likely seen it during pretraining.
It can then answer from memory even when the private context has been heavily noised. That makes a
privacy mechanism look better than it really is.

To remove this effect we built **Fantasy SQuAD** (Appendix C of the paper). We prompted GPT-4 to write
fictional encyclopedia entries and short stories about invented cities, events, and species, with
made-up names. Each passage comes with questions that can be answered only by reading that passage.

| | SQuAD | Fantasy SQuAD |
| --- | --- | --- |
| Source | Wikipedia articles | Fictional passages written by GPT-4 |
| Can the evaluator answer from memory? | Possibly | No, the facts are made up |
| Size | About 10,600 validation questions | 10 passages, 50 questions |
| Format | `context`, `question`, `answers` | Same as SQuAD |

The QA notebooks in `Fantasy_Dataset/` load Fantasy SQuAD. Because the format is the same, you can run
the same pipeline on the original SQuAD by changing the dataset line to
`load_dataset("squad", split="validation")`.

## Core mechanisms and STAMP

All mechanism implementations live in `functions.py`. At a high level:

* **Polar mechanism (direction-only vMF noise)**

  * Decomposes each embedding into radius + direction
  * Fixes radius, perturbs only the **direction** on the unit sphere using a von Mises–Fisher distribution
  * Decodes back to tokens via nearest neighbor search on normalized embeddings

* **Isotropic Laplace mechanism**

  * Computes an upper bound on the L1 sensitivity of embeddings
  * Adds i.i.d. Laplace noise to each embedding dimension
  * Decodes via nearest neighbor search

* **STAMP wrappers**

  * Given a context, a question, and a tokenizer/embedding table, STAMP:

    1. Assigns tokens to **four groups** based on sensitivity (NER/PII) and task-importance (embedding similarity to a task representation).
       In the notebooks, sensitive tokens are found with light rules (capitalized words, emails, and numbers). You can plug in a full NER tagger through the `ner_fn` argument.
    2. Maps each group to a **per-token ε**.
    3. Calls the chosen perturbation mechanism **per token** with the corresponding ε.
  * There are variants for:

    * Polar: `apply_stamp(...)`, `apply_stamp_allow_inf_polar(...)`
    * Laplace: `apply_stamp_laplace(...)`, `apply_stamp_allow_inf_laplace(...)`
  * ε=0 and ε=+∞ variants:

    * ε=0 can be interpreted as “drop token”
    * ε=+∞ as “keep token unchanged”

The notebooks show concrete usage examples of these functions.

---

## Notes on privacy and parameters

* **Metric-LDP:**
  Both Polar and Laplace mechanisms are calibrated to satisfy (ε)-metric-LDP with respect to a distance on embeddings (Euclidean for Laplace, geodesic/angle on the sphere for Polar).

* **Per-token ε vs. global ε:**
  STAMP operates in a **local** (per-token) privacy regime, with moderate-to-large ε values typical for DP text-rewriting work, because very small ε in high-dimensional embedding spaces destroys utility after decoding.

* **Group-wise budgets:**
  The default group profile is a simple monotone instantiation reflecting the ordering:

  > “Protect sensitive & task-unimportant tokens most; non-sensitive & task-important tokens least.”

  Users are free to change this profile; STAMP only requires a **relative ordering** and a base scale.

---

## Citation

```bibtex
@inproceedings{tian-etal-2026-stamp,
    title = "{STAMP}: Selective Task-Aware Mechanism for Text Privacy",
    author = "Tian, Fengwei and Bhattacharjee, Payel and Hanson, Heidi and Rubin, Geoffrey D and Lo, Joseph Y. and Tandon, Ravi",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.61/",
    doi = "10.18653/v1/2026.eacl-long.61",
    pages = "1319--1333"
}
```
