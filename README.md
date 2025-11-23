
![Banner](https://github.com/LittleHouse75/flatiron-resources/raw/main/NevitsBanner.png)

# Ethereum Scam Address Modeling

End-to-end address-level fraud modeling on Ethereum using a benchmark “synthetic” dataset, time-aware evaluation, and an external regulator dataset (California DFPI scam wallets).

---

## BLUF — What this project actually shows

Compared to the original “model zoo + maybe SGAN + synthetic augmentation” idea, this project ended up focusing on a ****single, well-tuned supervised pipeline**** and how it behaves under ****time and dataset shift****.

****Scope vs original plan****

- Started with: multiple classical models + a stretch goal of ATD-SGAN / synthetic data, plus a real-world regulator dataset.
- Ended with: a ****clean address-level feature pipeline****, ****XGBoost as the primary model****, and three evaluation views:
  1. ****Random address split**** (i.i.d. “ceiling” performance),
  2. ****Time-based split**** (past → future, showing drift),
  3. ****External DFPI dataset**** (transfer to a separate regulator-driven label source).
- The SGAN / synthetic-data piece was dropped as out-of-scope for the project timeline.

****Key results (headline metrics)****

- ****Random address split (same distribution, tuned XGBoost)****  
  - ROC AUC ≈ ****0.998****, Average Precision (AP) ≈ ****0.79****  
  - In a small high-risk band, roughly ****80% of flagged addresses are true scams****.  
  → On stable historical data, behavioral features are extremely powerful.

- ****Time-based split (train on past, test on future, tuned XGBoost)****  
  - AP in future window ≈ ****0.49–0.54****, ROC AUC ≈ ****0.90–0.91****  
  - At a precision-first operating point: Precision ≈ ****0.90+****, Recall ≈ ****0.25****  
  → The model still produces a very ****clean alert queue****, but ****misses many new scams**** as patterns drift over time.

- ****External DFPI evaluation (no retraining)****  
  - ROC AUC ≈ ****0.97****, AP ≈ ****0.90**** on a dataset built from ****California DFPI scam wallets**** plus background traffic.  
  - DFPI-listed scams tend to rank near the ****top of the score distribution****, with relatively few benign addresses mixed in.  
  → The model appears to learn ****transferable fraud signals****, not just quirks of the benchmark.

****Bottom line for a wallet provider****

- You get a concrete ****behavioral scoring pipeline**** (features + tuned XGBoost) that can:
  - Feed a ****warning banner**** before a send,
  - Drive a ****short, high-yield queue**** for investigators,
  - Maintain an ****internal, adaptive scam list**** that can be refreshed as new labels arrive.
- You also get a realistic sense of limits:
  - Random splits give an ****upper bound**** on performance.
  - Time-aware splits and DFPI tests show what happens in the real world, where ****drift and label delay**** are unavoidable.
  - Ongoing ****monitoring and retraining**** are mandatory if you want to keep catching new scams.

---

## 1. Project Overview

### Business problem

Public blockchains like Ethereum are full of fraud, but there is no standardized, shared “master list” of scam addresses. Wallet providers, exchanges, and analytics vendors maintain private lists based on:

- Proprietary labels,
- Ad-hoc rules and heuristics,
- Manual investigations.

These lists are expensive to maintain, incomplete, and slow to adapt to new scam patterns.

For a ****wallet provider****, this creates three main risks:

- Users unknowingly send funds to ****known scam addresses****.
- The provider faces ****reputational and compliance**** exposure.
- Fraud teams lose time on manual triage instead of higher-value investigative work.

### Goal

Build and evaluate a ****behavioral scam-scoring model**** at the ****address level**** that:

- Learns from historical labeled Ethereum transactions,
- Flags ****high-risk destinations**** before a transaction is sent,
- Helps internal teams maintain an up-to-date internal scam list without exposing proprietary labels.

### Modeling arc

The modeling story runs through three stages:

1. ****Benchmark → Address features****  
   - Start with an academic, labeled Ethereum ****transaction-level**** dataset.  
   - Aggregate to ****address-level behavioral features**** (degrees, value behavior, timing, gas, graph metrics).
   - Construct a binary ****Scam**** label at the address level.

2. ****Random vs time-based splits on the benchmark****  
   - ****Random address split****: classic i.i.d. setup; shows how well the model can separate scam vs non-scam when train/test share the same distribution.  
   - ****Time-based split****: train on ****earlier transactions****, test on ****later transactions****, with strict separation of past/future. This is closer to deployment reality and exposes concept drift.

3. ****External evaluation on California DFPI scam wallets****  
   - Pull scam addresses from the ****California Department of Financial Protection and Innovation (DFPI)**** crypto scam tracker.
   - Fetch on-chain transactions for these addresses and mix with background traffic.
   - Apply the tuned model trained on the benchmark ****without retraining**** and evaluate how well its scores align with DFPI-labeled scams.

---

## 2. Data Usage Guide

This section explains ****what data is used****, ****where it comes from****, and ****how it flows**** through the pipeline.

### 2.1 Benchmark Ethereum dataset (transactions)

****Source****

- “Labeled Transactions-based Dataset of Ethereum Network” (public academic benchmark from GitHub / associated paper).

****In this repo****

- Raw file expected at:  
  `data/Dataset.csv`  
  (Large raw files may be `.gitignored` depending on your setup.)

****Key columns (simplified)****

- Transaction-level fields: `hash`, `nonce`, `from_address`, `to_address`, `value`, `gas`, `gas_price`, `block_timestamp`, block metadata, etc.
- Scam-related fields: `from_scam`, `to_scam`, `from_category`, `to_category`, etc.

****How it’s used****

1. ****EDA & cleaning****  
   - Robust timestamp parsing into a consistent `block_timestamp_dt` (UTC).
   - Numeric coercion for `value`, `gas`, `gas_price`, etc.
   - Checks for timestamp-format leakage (e.g., scams vs non-scams using different string formats).

2. ****Splitting****  
   - ****Random split notebook****: sample addresses into train/val/test with stratification, ignoring time.
   - ****Time split notebook****: compute a time cutoff (e.g., 80% quantile of timestamps) and build ****past vs future**** transaction windows.

### 2.2 Engineered address-level features

****Generated files****

- Address-level tables, e.g.:  
  `data/address_features_random.parquet`  
  `data/address_features_time_train.parquet`  
  `data/address_features_time_test.parquet`  
  (Exact filenames may differ slightly; see notebooks and `src/featureeng.py`.)

****Feature families****

For each address (sender or receiver), features include:

- ****Degree / connectivity****
  - `in_degree`, `out_degree`, `all_degree`
  - `unique_in_degree`, `unique_out_degree`
- ****Amount behavior****
  - Incoming / outgoing `sum`, `mean`, `max`, `min` of `value`
- ****Temporal behavior****
  - Activity duration, transaction counts, inter-transaction gap statistics
  - Burstiness metrics (e.g., `max_gap / median_gap`)
  - Hour-of-day statistics (mean hour, hour entropy)
  - Recency and “last seen” relative to dataset start
- ****Gas behavior****
  - Average gas, average gas price
- ****Graph metrics****
  - Undirected clustering coefficient, etc.

****Labels****

- Constructed from the original scam-related fields.  
  - `Scam = 1` if an address appears in any scam-related field or category.  
  - `Scam = 0` otherwise.

****How to regenerate****

- The feature engineering logic lives in `src/featureeng.py`.
- An example entry point is:

  ```python
  from src import featureeng
  
  addr_df = featureeng.engineer_features(
      transaction_df=df,
      target_addresses=target_addresses,
      addr_labels=addr_labels,
      global_start=global_start,
      split_name="Random Split"
  )

* Notebooks call into these helpers so the exact feature recipe stays consistent across experiments.

⠀
### 2.3 DFPI external evaluation dataset

**Source**
* California **Department of Financial Protection and Innovation (DFPI)** Crypto Scam Tracker.

⠀
**Pipeline (high-level)**
1. Scrape or download **scam wallet addresses** from the DFPI website.
2. Fetch **on-chain transactions** for these scam addresses plus a sample of non-listed addresses as background.
3. Run them through the **same feature engineering functions** as the benchmark data, producing a DFPI address-level table.
4. Load the **tuned model** (pickled from the benchmark training) and generate scam probabilities for each address.
5. Evaluate AP, ROC AUC, and precision/recall at relevant thresholds.

⠀
**In this repo**
* The exact DFPI data files are likely under data/dfpi_* (CSV/Parquet), plus an **external evaluation notebook** (e.g. 04_DFPI_External_Eval.ipynb) that:
  * loads the trained model,
  * builds or loads DFPI address features,
  * computes and visualizes performance.


⸻

### 3. Modeling and Evaluation Summary

* **Models tried**
  * Logistic Regression (with class_weight=“balanced”)
  * Random Forest
  * ExtraTrees
  * XGBoost (primary focus)
  * MLPClassifier (exploratory)
* **Evaluation metrics**
  * Primary: **Average Precision (AP)**, due to heavy class imbalance.
  * Secondary: ROC AUC, precision, recall, F1, confusion matrices, calibration curves.
  * For DFPI external evaluation: AP, ROC AUC, and precision/recall at a chosen operating threshold.
* **Interpretability**
  * Feature importance and SHAP plots (for XGBoost) to understand which behavioral features drive scam predictions (e.g., bursts of activity, value patterns, timing, degree structure).
⠀