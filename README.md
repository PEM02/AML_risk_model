# AML Risk Model — Suspicious Account Detection

An end-to-end machine learning project that identifies potentially suspicious
(money-laundering) accounts from raw transaction data. The work is split into
three notebooks — exploratory analysis, feature engineering, and modeling —
each documenting not just *what* was done but *why*, including the data-quality
issues and generator artifacts that shape every downstream decision.

## Problem

Given account-level metadata and a full transfer log, the goal is to produce a
per-account **risk score** for money-laundering activity. The two laundering
typologies present in the data are graph patterns:

- **`fan_in`** — many distinct senders converging small amounts on one account
  (structuring / smurfing).
- **`cycle`** — funds traversing a chain of accounts and returning to the origin
  (layering).

The prediction unit is the **account**, and the target is a moderately
imbalanced binary label (~16.85% positive).

## Data

The dataset is **synthetic**, generated with
[AMLSim](https://github.com/IBM/AMLSim), and consists of three tables:

| File | Rows | Content |
|---|---|---|
| `accounts.csv` | 10,000 | One row per account: balance, behavior profile, fraud label |
| `alerts.csv` | 1,719 | The subset of transactions belonging to a known laundering pattern |
| `transactions.csv` | 1,323,234 | The complete account-to-account transfer log |

> **The raw CSVs are not included in this repository.** They are excluded via
> `.gitignore`. To reproduce the results, generate an equivalent dataset with
> AMLSim and place the three files in the project root (see the loading cells at
> the top of notebook 01).

## Repository structure

```
.
├── 01_EDA_AML.ipynb                 # Data quality, univariate/bivariate EDA, network verification
├── 02_Feature_Engineering_AML.ipynb # Leakage-safe split, winsorization, 38 account-level features
├── 03_Modeling_AML.ipynb            # Logistic Regression vs XGBoost, ablation, SHAP interpretation
├── environment.yml                  # Conda environment
├── .gitignore
└── README.md
```

## How to run

```bash
# 1. Create and activate the environment
conda env create -f environment.yml
conda activate aml

# 2. Place the three AMLSim CSVs in the project root, then launch Jupyter
jupyter lab   # or: jupyter notebook
```

Run the notebooks in order (`01` → `02` → `03`). Notebook 02 exports the
processed features and the train/test split to `data_processed/`, which
notebook 03 consumes — this guarantees a consistent split and reproducible
results.

## Approach

**1. Exploratory analysis (`01`)** — verifies data integrity (missing values,
duplicates, referential integrity) and characterizes the problem. Two findings
drive everything after: a **32-bit integer-overflow defect** in transaction
amounts (the maximum value matches exactly the signed int32 ceiling ÷ 100), and
the fact that laundering transactions occupy an amount band that never overlaps
with normal traffic — a fingerprint of the generator rather than real behavior.

**2. Feature engineering (`02`)** — builds 38 account-level features across
seven families (volume, network degree, flow, counterparty repetition,
reciprocity, temporal, balance-relative). The design principle: the typologies
are *structural*, so the features prioritize **with whom** and **how** an account
transacts over **how much**. Leakage is prevented explicitly — the train/test
split happens *before* feature construction, and the winsorization threshold is
computed on training data only.

**3. Modeling (`03`)** — trains an interpretable Logistic Regression baseline and
an XGBoost model, then audits the result. An ablation isolates genuine
behavioral signal from generator artifacts, and SHAP explains individual
predictions.

## Key findings

- **A trivial rule beats the full model — and that's the point.** Flagging any
  account that ever sent a transaction below \$20 achieves ~90% recall with no
  model at all. This exposes the amount artifact: magnitude features detect the
  generator's injection band, not laundering behavior, and would not generalize
  to production.
- **The structural signal is real.** After removing all amount-magnitude
  features *and* the generator's behavior-profile metadata, an XGBoost model
  still reaches **ROC-AUC ≈ 0.94**, relying on network structure — distinct
  counterparties, reciprocity, and burst timing — exactly the features designed
  to capture `fan_in` and `cycle`.
- **`in_degree` (distinct senders) is the strongest single signal**, consistent
  with `fan_in` concentrating its anomaly on the receiving account.
- **PR-AUC and Precision@K are prioritized over ROC-AUC**, because in a realistic
  AML setting (<1% prevalence) the operational constraint is how many genuine
  cases a finite analyst team can capture in the top-K riskiest accounts.

## Limitations

- Class imbalance (16.85%) is far above real-world AML rates (<1%), so metrics
  here likely overstate production performance.
- Only two typologies (`cycle`, `fan_in`) and a single channel (transfers only)
  are represented — no cash, `fan_out`, or scatter-gather patterns.
- The amount-magnitude artifact means any model using those features on this
  dataset produces inflated, non-transferable metrics.

## Possible next steps

- Re-evaluate at realistic prevalence by downsampling the positive class to ~1%.
- Add second-order graph features (PageRank, betweenness, community detection)
  or a Graph Neural Network that learns directly from the topology.
- Calibrate the decision threshold to the investigative team's actual capacity
  and optimize for Precision@K.
- Use a temporal split (train on earlier periods, test on later) to mirror
  deployment and enable drift detection.
