
![Banner](https://github.com/LittleHouse75/flatiron-resources/raw/main/NevitsBanner.png)

# Ethereum Scam Address Modeling

End-to-end address-level fraud modeling on Ethereum using a benchmark academic dataset, time-aware evaluation, and an external regulator dataset (California DFPI scam wallets).

---

## BLUF — What this project actually shows

Compared to the early idea of “model zoo + SGAN + synthetic augmentation,” this project ultimately focused on a **single, well-tuned supervised pipeline** and how it behaves under **time drift** and **dataset shift**.

### Scope vs the original plan

- **Originally:** multiple classical models, exploratory SGAN work, synthetic data generation, and a regulator dataset.
- **Final project:**  
  - A clean **address-level feature pipeline**,  
  - **XGBoost** as the main model,  
  - Three evaluation views that tell one coherent story:
    1. **Random address split** (i.i.d. “ceiling” conditions)  
    2. **Time-based split** (past → future, showing real drift)  
    3. **External DFPI evaluation** (transfer to a regulator dataset)

The SGAN portion was dropped as out-of-scope for a fixed project timeline.

### Key results (headline metrics)

#### **Random address split (i.i.d., tuned XGBoost)**
- ROC AUC ≈ **0.998**  
- Average Precision (AP) ≈ **0.79**  
- In a tight high-risk band, around **80% of alerts** are true scams  
→ When train/test share the same distribution, behavioral features are extremely powerful.

#### **Time-based split (train on past, test on future)**
- ROC AUC ≈ **0.90–0.91**  
- Future-window AP ≈ **0.49–0.54**  
- Precision-first setting: **~0.90 precision**, **~0.25 recall**  
→ The model still yields a *clean* alert queue, but misses many **new** scams as patterns drift.

#### **External DFPI evaluation (no retraining)**
- ROC AUC ≈ **0.97–0.98**  
- Average Precision ≈ **0.11** (vs DFPI base rate ≈ **0.02%**)  
- Known DFPI scams cluster in the **extreme right tail** of predicted risk  
→ The model learned **transferable behavioral signals**, not just benchmark quirks.

### Bottom line for a wallet provider

You get a concrete **behavioral scoring pipeline** (address features + tuned XGBoost) that can:

- Power an **in-app warning banner** before a transaction is sent  
- Drive a **short, high-yield triage queue** for analysts  
- Maintain an **internal, continuously refreshed scam list**

You also get a realistic sense of limits:

- Random splits are an **upper bound**.  
- Time splits + DFPI tests show what happens in the real world where **drift** and **label delay** exist.  
- **Retraining and monitoring** are essential if you want to keep catching new scam patterns.

---

## 1. Project Overview

### Business problem

Public blockchains like Ethereum are full of fraud, but there is no standardized, shared “master list” of malicious addresses. Wallet providers and exchanges maintain their own private lists based on:

- Proprietary labels  
- Ad-hoc rules and heuristics  
- Manual investigations  

These lists are incomplete, expensive, and slow to adapt to evolving scams.

This creates serious risk:

- Users unknowingly send to **known scam addresses**  
- Reputational and regulatory exposure  
- Fraud teams spend hours in low-value manual triage

### Goal

Build and evaluate a **behavioral scam-scoring model** at the **address level** that:

- Learns from historical labeled transactions  
- Flags **high-risk destinations** before users send funds  
- Helps maintain an internal, adaptive scam list without exposing proprietary labels  

### Modeling arc

1. **Benchmark → Address features**  
   - Start with a public, labeled Ethereum **transaction-level** dataset  
   - Aggregate to **address-level behavioral features**  
   - Construct a binary **Scam** label at the address level  

2. **Random vs. time-based splits on the benchmark**  
   - **Random address split** → ideal i.i.d. scenario  
   - **Time-based split** → train on past, test on future, revealing drift  

3. **External evaluation on California DFPI scam wallets**  
   - Pull DFPI-labeled scam addresses  
   - Fetch their on-chain history + background traffic  
   - Apply the tuned model **without retraining**  
   - Check whether DFPI scams still surface as high-risk  

---

## 2. Data Usage Guide

This section explains **what data is used**, **where it comes from**, and **how it flows** through the pipeline.

### 2.1 Benchmark Ethereum dataset (transactions)

**Source**  
- “Labeled Transactions-based Dataset of Ethereum Network” (public academic dataset)

**In this repo**  
- Expected raw file: `data/Dataset.csv`

**Key columns**  
- Core transaction fields: `hash`, `nonce`, `from_address`, `to_address`, `value`, `gas`, `gas_price`, `block_timestamp`, etc.  
- Scam fields: `from_scam`, `to_scam`, `from_category`, `to_category`

**How it’s used**  
1. **EDA & cleaning**  
   - Timestamp normalization  
   - Numeric coercion  
   - Checks for format-based leakage  

2. **Splitting**  
   - **Random split**: stratified address-level split, ignoring time  
   - **Time split**: compute an 80% timestamp cutoff → build past vs future transaction windows  

### 2.2 Address-level engineered features

**Generated files**  
Examples:  
- `address_features_random.parquet`  
- `address_features_time_train.parquet`  
- `address_features_time_test.parquet`

**Feature families**

- **Degree / connectivity**  
  `in_degree`, `out_degree`, `unique_in_degree`, `unique_out_degree`, etc.

- **Amount behavior**  
  Sums, means, max/min, inbound vs outbound value patterns  

- **Temporal behavior**  
  Duration, gaps, burst metrics, hour-of-day statistics, last-seen recency  

- **Gas behavior**  
  Average gas, gas price patterns  

- **Graph metrics**  
  Local clustering, undirected degree  

**Labeling**  
- `Scam = 1` if an address appears in any scam field/category  
- `Scam = 0` otherwise  

**How to regenerate**  
Feature engineering lives in `src/featureeng.py`.

Example entry point:

```python
from src import featureeng

addr_df = featureeng.engineer_features(
    transaction_df=df,
    target_addresses=target_addresses,
    addr_labels=addr_labels,
    global_start=global_start,
    split_name="Random Split"
)
```

### 2.3 DFPI external evaluation dataset

**Source**
* California Department of Financial Protection and Innovation (DFPI) — Crypto Scam Tracker

⠀
**Pipeline**
1. Download DFPI scam wallet addresses
2. Fetch on-chain transactions for these addresses + background traffic
3. Run through the same address-level feature pipeline
4. Load the tuned model trained on the benchmark
5. Score addresses and evaluate **ROC AUC**, **AP**, and **precision/recall**

⠀
**In this repo**
* DFPI data under data/dfpi_*
* External evaluation notebook: 04_DFPI_ExternalEval.ipynb

⠀
⸻

### 3. Modeling and Evaluation Summary

**Models examined**
* Logistic Regression
* Random Forest
* ExtraTrees
* XGBoost (primary)
* MLPClassifier (exploratory)

⠀
**Metrics**
* Primary: **Average Precision (AP)**
* Secondary: ROC AUC, precision, recall, F1, confusion matrices, calibration curves
* DFPI external: AP, ROC AUC, and precision/recall at meaningful thresholds

⠀
**Interpretability**
* Feature importance + SHAP (XGBoost)
* Behavioral drivers: burstiness, degree, temporal patterns, value flows