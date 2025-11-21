# STAMP: Selective Task-Aware Mechanism for Text Privacy

This repository contains the **reference implementation** for the paper

> **STAMP: Selective Task-Aware Mechanism for Text Privacy**
> (Anonymous submission)

STAMP is a framework for **task-aware, sensitivity-aware privacy budgeting** for text under local differential privacy (LDP). It combines:

* A **2×2 grouping** of tokens by *sensitivity* (e.g., PII) and *task importance*
* A **Polar mechanism** that adds noise only to the **direction** of token embeddings (on the unit sphere)
* Standard **metric-LDP** guarantees per token, lifted to sequences via parallel composition

We compare STAMP against a **uniform-budget Laplace mechanism**.

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
  Synthetic SQuAD-style QA dataset (same JSON structure as SQuAD: `data -> paragraphs -> {context, qas}`) used in some experiments. (To make sure the models have not learned these sentences from internal memory.)

* Notebooks (for experiments and figures)

---

## Datasets

The notebooks use standard HuggingFace datasets plus one synthetic file:

* **AG News**: loaded via `datasets.load_dataset("ag_news")`
* **Yelp**: loaded via `datasets.load_dataset("yelp")`
* **SQuAD / FantasySQuAD**:

  * `fantasy_squad.json` from this repo, which follows the SQuAD format and can be loaded via:

   

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
    2. Maps each group to a **per-token ε** (e.g., (\epsilon_{G1}:\epsilon_{G2}:\epsilon_{G3}:\epsilon_{G4} = 2:1:4:3) × base (\bar{\epsilon})).
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
  Both Polar and Laplace mechanisms are calibrated to satisfy (\epsilon)-metric-LDP with respect to a distance on embeddings (Euclidean for Laplace, geodesic/angle on the sphere for Polar).

* **Per-token ε vs. global ε:**
  STAMP operates in a **local** (per-token) privacy regime, with moderate-to-large ε values typical for DP text-rewriting work, because very small ε in high-dimensional embedding spaces destroys utility after decoding.

* **Group-wise budgets:**
  The default group profile:

  [
  \epsilon_{G1} : \epsilon_{G2} : \epsilon_{G3} : \epsilon_{G4}
  = 2 : 1 : 4 : 3 \quad (\text{all scaled by } \bar{\epsilon})
  ]

  is a simple monotone instantiation reflecting the ordering:

  > “Protect sensitive & task-unimportant tokens most; non-sensitive & task-important tokens least.”

  Users are free to change this profile; STAMP only requires a **relative ordering** and a base scale (\bar{\epsilon}).

## Contact / issues

This repository is provided as an **anonymous artifact** for peer review.
If you encounter reproducibility issues or bugs while evaluating the paper, please include:

* The environment (OS, Python, CUDA, PyTorch versions)
* The exact notebook / script and cell that failed
* Any error messages

in your report so we can address them in the camera-ready version.
