"""
Address-level feature engineering utilities for the Ethereum scam detection project.

This module provides:
  - Timestamp normalization and basic time-of-day features
  - Global address-level scam labels
  - Modular helpers to build per-address behavioral features
"""

import os
import sys

import numpy as np
import pandas as pd
from IPython.display import display

import src.utilities as util

# Make the project root importable when running notebooks from subfolders
sys.path.append(os.path.abspath(".."))


def normalize_timestamps(
    df: pd.DataFrame,
    ts_col: str = "block_timestamp",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Normalize raw timestamp strings into a UTC datetime and add
    simple time-of-day / weekday features.

    Mutates df in-place **and** returns it.
    Adds columns:
      - f"{ts_col}_raw"  : original values as string
      - f"{ts_col}_std"  : cleaned, normalized string
      - f"{ts_col}_dt"   : parsed datetime (UTC)
      - "hour"           : hour-of-day (0–23) from the parsed datetime
      - "weekday"        : weekday name from the parsed datetime
    """

    raw_col = f"{ts_col}_raw"
    std_col = f"{ts_col}_std"
    dt_col = f"{ts_col}_dt"

    if verbose:
        util.print_sub_heading("Timestamp normalization")
        print(
            f"Converting `{ts_col}` into a consistent UTC datetime and "
            f"adding derived features (hour, weekday)."
        )
        print("Value types in timestamp column:")
        print(df[ts_col].apply(type).value_counts())
        print("Number of missing raw timestamp values:", df[ts_col].isna().sum())

    # 1. Preserve original raw text for audit/debugging
    df[raw_col] = df[ts_col].astype(str)

    # 2. Normalize unicode and convert to a clean working string
    raw_ts = df[raw_col].str.normalize("NFKC")

    # 3. Clean suffix patterns and normalize formatting
    clean_ts = (
        raw_ts.str.replace(" UTC", "", regex=False)
        .str.replace(" UTC+0000", "", regex=False)
        .str.replace("+0000 UTC", "", regex=False)
        .str.replace(" Z", "", regex=False)
        .str.replace("+00:00", "", regex=False)
        .str.strip()
    )
    df[std_col] = clean_ts

    # 4. Parse to datetime (UTC)
    df[dt_col] = pd.to_datetime(
        df[std_col],
        utc=True,
        errors="coerce",
    )

    # 5. Derive time-of-day and weekday features from the parsed datetime
    df["hour"] = df[dt_col].dt.hour
    df["weekday"] = df[dt_col].dt.day_name()

    if verbose:
        print("Parsed timestamps (non-null):", df[dt_col].notna().sum())
        print("Unparseable timestamps:", df[dt_col].isna().sum())

    return df

def build_address_labels(
    df: pd.DataFrame,
    from_scam_col: str = "from_scam",
    to_scam_col: str = "to_scam",
    from_cat_col: str = "from_category",
    to_cat_col: str = "to_category",
    from_addr_col: str = "from_address",
    to_addr_col: str = "to_address",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build global address-level Scam labels (0/1) from the full transaction df.

    Returns a DataFrame with:
      - 'Address'
      - 'Scam'  (1 if ever involved in scam-like activity)
    """

    def is_scam_category(x):
        if pd.isna(x):
            return False
        x = str(x).lower()
        return ("scam" in x) or ("fraud" in x) or ("phish" in x)

    # One global address table: each address gets a single Scam label (0/1)
    if verbose:
        util.print_sub_heading("Building Address-Level Scam Labels")

    # Flag scam-like transactions (used only to derive address-level labels)
    from_is_scam = (
        df[from_scam_col].fillna(0).astype(int)
        | df[from_cat_col].apply(is_scam_category)
    )
    to_is_scam = (
        df[to_scam_col].fillna(0).astype(int)
        | df[to_cat_col].apply(is_scam_category)
    )

    scam_addresses = pd.Index(
        pd.concat(
            [
                df.loc[from_is_scam, from_addr_col],
                df.loc[to_is_scam, to_addr_col],
            ]
        )
    ).dropna().unique()

    # All unique addresses seen anywhere in the dataset
    all_addresses = pd.Index(
        pd.concat([df[from_addr_col], df[to_addr_col]])
    ).dropna().unique()

    addr_df = pd.DataFrame({"Address": all_addresses})
    addr_df["Scam"] = addr_df["Address"].isin(scam_addresses).astype(int)

    if verbose:
        print("Total addresses:", len(addr_df))
        print("Total scam addresses:", int(addr_df["Scam"].sum()))

    return addr_df

# ============================================================
# Address-Level Feature Engineering (Modular)
# ============================================================

# ------------------------------------------------------------
# 1. TIMESTAMP BLOCK
# ------------------------------------------------------------
def _add_timestamps(tx: pd.DataFrame, global_start: pd.Timestamp) -> pd.DataFrame:
    """
    Add a global time axis for this split.

    Parameters
    ----------
    tx : pd.DataFrame
        Transaction-level data that already includes a parsed
        `block_timestamp_dt` (UTC) column.
    global_start : pd.Timestamp
        Global reference start time (e.g., min over full dataset)
        used to compute seconds-since-start.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with:
          - "block_timestamp_raw_string": original timestamp as string
          - "ts_seconds": seconds since `global_start` (float)
    """
    if "block_timestamp_dt" not in tx.columns:
        raise ValueError(
            "block_timestamp_dt missing. Run timestamp normalization first."
        )

    # Preserve original string and add a global "seconds since start" time axis
    tx["block_timestamp_raw_string"] = tx["block_timestamp"].astype(str)
    tx["ts_seconds"] = (tx["block_timestamp_dt"] - global_start).dt.total_seconds()
    return tx

# ------------------------------------------------------------
# 2. NUMERIC CLEANING
# ------------------------------------------------------------
def _clean_numeric_fields(tx: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Coerce core numeric transaction fields into clean numeric columns.

    Parameters
    ----------
    tx : pd.DataFrame
        Transaction-level data with raw numeric-like columns
        "value", "gas", and "gas_price".
    verbose : bool, default True
        If True, prints simple diagnostics about invalid values.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with "value", "gas", and "gas_price" converted
        to numeric dtype and any invalid entries filled with 0.
    """
    if verbose:
        util.print_sub_heading("Numeric Column Diagnostics")

    for col in ["value", "gas", "gas_price"]:
        raw = tx[col].astype(str)
        bad_mask = pd.to_numeric(raw, errors="coerce").isna()
        if verbose:
            print(f"{col}: invalid={bad_mask.sum()}")
        tx[col] = pd.to_numeric(raw, errors="coerce").fillna(0)

    if verbose:
        print("Numeric cleaning complete. Rows retained:", len(tx))
    return tx

# ------------------------------------------------------------
# 3. DEGREE + AMOUNT FEATURES
# ------------------------------------------------------------
def _compute_degree_amount_features(tx: pd.DataFrame):
    """
    Compute degree (in/out) and basic amount statistics per address.

    Parameters
    ----------
    tx : pd.DataFrame
        Transaction-level data with "from_address", "to_address",
        and cleaned numeric "value".

    Returns
    -------
    tuple
        (
          in_degree          : pd.Series (# tx received per to_address),
          out_degree         : pd.Series (# tx sent per from_address),
          unique_in_degree   : pd.Series (# unique senders per to_address),
          unique_out_degree  : pd.Series (# unique recipients per from_address),
          incoming_amounts   : pd.DataFrame with mean/sum/max/min value per to_address,
          outgoing_amounts   : pd.DataFrame with mean/sum/max/min value per from_address,
        )
    """
    # Degrees
    in_degree = tx.groupby("to_address").size().rename("in_degree")
    out_degree = tx.groupby("from_address").size().rename("out_degree")

    unique_in_degree = (
        tx.groupby("to_address")["from_address"]
        .nunique()
        .rename("unique_in_degree")
    )

    unique_out_degree = (
        tx.groupby("from_address")["to_address"]
        .nunique()
        .rename("unique_out_degree")
    )

    # Amounts
    incoming_vals = tx[tx["to_address"].notna()].copy()
    incoming_vals["value"] = incoming_vals["value"].replace({0: np.nan})
    incoming_amounts = (
        incoming_vals.groupby("to_address")["value"]
        .agg(["mean", "sum", "max", "min"])
        .rename(
            columns={
                "mean": "Avg amount incoming",
                "sum": "Total amount incoming",
                "max": "Max amount incoming",
                "min": "Min amount incoming",
            }
        )
    )

    outgoing_vals = tx[tx["from_address"].notna()].copy()
    outgoing_vals["value"] = outgoing_vals["value"].replace({0: np.nan})
    outgoing_amounts = (
        outgoing_vals.groupby("from_address")["value"]
        .agg(["mean", "sum", "max", "min"])
        .rename(
            columns={
                "mean": "Avg amount outgoing",
                "sum": "Total amount outgoing",
                "max": "Max amount outgoing",
                "min": "Min amount outgoing",
            }
        )
    )

    return (
        in_degree,
        out_degree,
        unique_in_degree,
        unique_out_degree,
        incoming_amounts,
        outgoing_amounts,
    )

# ------------------------------------------------------------
# 4. TIME-BASED METRICS + ADVANCED TEMPORAL FEATURES
# ------------------------------------------------------------
def _compute_time_metrics(group: pd.DataFrame) -> pd.Series:
    """
    Summarize basic timing structure within a single address history.

    Parameters
    ----------
    group : pd.DataFrame
        All transactions for a single address in long format,
        expected to include a "ts_seconds" column.

    Returns
    -------
    pd.Series
        A series with:
          - "Active Duration"     : last_ts - first_ts
          - "Total Tx Time"       : sum of gaps between consecutive tx
          - "Mean time interval"  : mean gap between tx
          - "Max time interval"   : max gap between tx
          - "Min time interval"   : min gap between tx
        All values are in seconds.
    """
    times = np.sort(group["ts_seconds"].dropna().values)

    if len(times) <= 1:
        return pd.Series(
            {
                "Active Duration": 0.0,
                "Total Tx Time": 0.0,
                "Mean time interval": 0.0,
                "Max time interval": 0.0,
                "Min time interval": 0.0,
            }
        )

    gaps = np.diff(times)
    return pd.Series(
        {
            "Active Duration": float(times[-1] - times[0]),
            "Total Tx Time": float(gaps.sum()),  # TRUE definition
            "Mean time interval": float(gaps.mean()),
            "Max time interval": float(gaps.max()),
            "Min time interval": float(gaps.min()),
        }
    )

def _compute_advanced_time_features(tx: pd.DataFrame, verbose: bool = True):
    """
    Build advanced temporal and behavioral features in a long format.

    Parameters
    ----------
    tx : pd.DataFrame
        Transaction-level data for the current split. Must contain:
          - "from_address", "to_address"
          - "ts_seconds"
          - "gas", "gas_price"
          - "block_timestamp_dt" (UTC datetime)
    verbose : bool, default True
        If True, prints a high-level heading for this block.

    Returns
    -------
    dict
        Dictionary of intermediate per-address objects:
          - "long"         : long-format DataFrame with one row per (address, tx)
          - "avg_in"       : mean ts_seconds for incoming tx per address
          - "avg_out"      : mean ts_seconds for outgoing tx per address
          - "time_metrics" : DataFrame of active duration / gap stats per address
          - "burstiness"   : per-address burstiness score
          - "hour_df"      : DataFrame with "Hour mean" and "Hour entropy"
          - "incoming_ct"  : # incoming tx per address
          - "outgoing_ct"  : # outgoing tx per address
          - "last_seen"    : last ts_seconds each address appears
          - "dataset_end"  : max ts_seconds in this split (scalar)
    """
    if verbose:
        util.print_sub_heading("Advanced Temporal Behavior Features")

    # Long format IN
    long_in = tx[
        ["to_address", "ts_seconds", "gas", "gas_price", "block_timestamp_dt"]
    ].rename(
        columns={
            "to_address": "Address",
            "block_timestamp_dt": "timestamp",
        }
    )
    long_in["direction"] = "in"

    # Long format OUT
    long_out = tx[
        ["from_address", "ts_seconds", "gas", "gas_price", "block_timestamp_dt"]
    ].rename(
        columns={
            "from_address": "Address",
            "block_timestamp_dt": "timestamp",
        }
    )
    long_out["direction"] = "out"

    long_advanced = (
        pd.concat([long_in, long_out], ignore_index=True)
        .dropna(subset=["Address"])
    )
    long_advanced["hour"] = long_advanced["timestamp"].dt.hour

    # Avg times (incoming / outgoing)
    avg_in = (
        long_advanced[long_advanced["direction"] == "in"]
        .groupby("Address")["ts_seconds"]
        .mean()
        .rename("Avg time incoming")
    )
    avg_out = (
        long_advanced[long_advanced["direction"] == "out"]
        .groupby("Address")["ts_seconds"]
        .mean()
        .rename("Avg time outgoing")
    )

    # Core time metrics
    time_metrics = long_advanced.groupby("Address").apply(_compute_time_metrics)

    # Burstiness
    def burst(g: pd.DataFrame) -> pd.Series:
        t = np.sort(g["ts_seconds"].dropna().values)
        if len(t) <= 2:
            return pd.Series({"Burstiness": 0.0})
        gaps = np.diff(t)
        med = np.median(gaps)
        return pd.Series(
            {"Burstiness": float(gaps.max() / med if med else gaps.max())}
        )

    burstiness = long_advanced.groupby("Address").apply(burst)

    # Hour mean + entropy
    records = []
    for addr, g in long_advanced.groupby("Address"):
        hrs = g["hour"].dropna().astype(int).values
        if len(hrs) == 0:
            records.append((addr, 0.0, 0.0))
            continue
        hist = np.bincount(hrs, minlength=24)
        probs = hist / hist.sum()
        nz = probs[probs > 0]
        entropy = float(-(nz * np.log2(nz)).sum())
        records.append((addr, float(hrs.mean()), entropy))

    hour_df = pd.DataFrame(
        records, columns=["Address", "Hour mean", "Hour entropy"]
    ).set_index("Address")

    # Incoming/outgoing counts
    inc_ct = (
        long_advanced[long_advanced["direction"] == "in"]
        .groupby("Address")
        .size()
        .rename("Incoming count")
    )
    out_ct = (
        long_advanced[long_advanced["direction"] == "out"]
        .groupby("Address")
        .size()
        .rename("Outgoing count")
    )

    # Recency anchor
    dataset_end = tx["ts_seconds"].max()
    last_seen = (
        long_advanced.groupby("Address")["ts_seconds"]
        .max()
        .rename("Last seen")
    )

    return {
        "long": long_advanced,
        "avg_in": avg_in,
        "avg_out": avg_out,
        "time_metrics": time_metrics,
        "burstiness": burstiness,
        "hour_df": hour_df,
        "incoming_ct": inc_ct,
        "outgoing_ct": out_ct,
        "last_seen": last_seen,
        "dataset_end": dataset_end,
    }

# ------------------------------------------------------------
# 5. GAS FEATURES
# ------------------------------------------------------------
def _compute_gas_features(long: pd.DataFrame, verbose: bool = True):
    """
    Compute average gas usage per address from a long-format table.

    Parameters
    ----------
    long : pd.DataFrame
        Long-format DataFrame with columns:
          - "Address"
          - "gas"
          - "gas_price"
    verbose : bool, default True
        If True, prints a short heading for this block.

    Returns
    -------
    tuple
        (
          avg_price : pd.Series (mean gas_price per address),
          avg_limit : pd.Series (mean gas per address),
        )
    """
    if verbose:
        util.print_sub_heading("Gas-Based Features")

    gas = long.groupby("Address")
    avg_price = gas["gas_price"].mean().rename("Avg gas price")
    avg_limit = gas["gas"].mean().rename("Avg gas limit")
    return avg_price, avg_limit

# ------------------------------------------------------------
# MAIN ORCHESTRATOR
# ------------------------------------------------------------
def engineer_address_features(
    transaction_df: pd.DataFrame,
    target_addresses,
    addr_labels: pd.DataFrame,
    global_start: pd.Timestamp,
    split_label: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a full per-address feature table for a given split of transactions.

    The pipeline:
      1) Adds a global time axis (seconds since `global_start`)
      2) Cleans core numeric fields (value, gas, gas_price)
      3) Computes degree and amount statistics (in/out counts, value stats)
      4) Derives temporal behavior (gaps, burstiness, hour-of-day, recency)
      5) Adds gas usage features
      6) Aligns to the provided target address list and attaches Scam labels
      7) Applies a consistent fill strategy for missing values

    Parameters
    ----------
    transaction_df : pd.DataFrame
        Raw transaction rows for the current split only
        (e.g., past window, future window, or full dataset).
    target_addresses : iterable
        Addresses that should appear as the index of the output feature table.
        Any address not in this list will be dropped; any missing will be added
        with filled features.
    addr_labels : pd.DataFrame
        Global address-level labels built once from the full dataset, with:
          - "Address"
          - "Scam" (0/1)
    global_start : pd.Timestamp
        Reference start time for the entire project (e.g., min over
        full df["block_timestamp_dt"]). Used to make ts_seconds comparable
        across splits.
    split_label : str
        Human-readable label for logging (e.g., "All Addresses",
        "Past Window — Time Split", "Random Split — Train+Val").
    verbose : bool, default True
        If True, prints headings, diagnostics, and a small sample of
        the resulting feature table.

    Returns
    -------
    pd.DataFrame
        Address-level feature matrix indexed by "Address", including:
          - Structural features (degrees, counts)
          - Value statistics (incoming/outgoing means, sums, mins/maxes)
          - Temporal features (durations, gaps, burstiness, hour stats, recency)
          - Gas usage features
          - Target column "Scam" (0/1)
    """

    split_label = split_label or "Addresses"
    if verbose:
        util.print_heading(
            f"Preparing Transaction Data for Feature Engineering — {split_label}"
        )

    tx = transaction_df.copy()
    if verbose:
        print("Initial row count:", len(tx))

    # Step 1: Timestamps
    tx = _add_timestamps(tx, global_start)

    # Step 2: Numeric cleaning
    tx = _clean_numeric_fields(tx, verbose=verbose)

    # Step 3: Degree + amount features (per-address counts and value stats)
    (
        in_degree,
        out_degree,
        unique_in,
        unique_out,
        inc_amts,
        out_amts,
    ) = _compute_degree_amount_features(tx)

    # Base index: all addresses present in this split
    all_addrs = pd.concat([tx["from_address"], tx["to_address"]]).dropna().unique()
    features = pd.DataFrame(index=all_addrs)
    features.index.name = "Address"

    # Join degree-related features
    features = (
        features.join(in_degree, how="left")
        .join(out_degree, how="left")
        .join(unique_in, how="left")
        .join(unique_out, how="left")
    )

    # Tx count
    features["Tx count"] = (
        features["in_degree"].fillna(0).astype(int)
        + features["out_degree"].fillna(0).astype(int)
    )

    # Join amount-related features
    features = (
        features.join(inc_amts, how="left")
        .join(out_amts, how="left")
    )

    # Step 4: Advanced temporal block (time gaps, burstiness, hour-of-day, recency)
    T = _compute_advanced_time_features(tx, verbose=verbose)

    features = (
        features.join(T["avg_in"], how="left")
        .join(T["avg_out"], how="left")
        .join(T["time_metrics"], how="left")
        .join(T["burstiness"], how="left")
        .join(T["hour_df"], how="left")
        .join(T["incoming_ct"], how="left")
        .join(T["outgoing_ct"], how="left")
        .join(T["last_seen"], how="left")  # <- critical to avoid KeyError
    )

    # Density
    duration = features["Active Duration"].fillna(0.0)
    features["Activity Density"] = features["Tx count"] / (duration + 1.0)

    # In/Out ratio
    inc = features["Incoming count"].fillna(0)
    out = features["Outgoing count"].fillna(0)
    features["In/Out Ratio"] = (inc + 1) / (out + 1)

    # Recency
    # Distinguish "never seen" from "seen at time 0" by filling NaN with -1.0:
    #   - close to 0 for scaling
    #   - negative so trees can easily separate it from valid timestamps.
    features["Last seen"] = features["Last seen"].fillna(-1.0)
    features["Recency"] = T["dataset_end"] - features["Last seen"]

    # Step 5: Gas features
    avg_price, avg_limit = _compute_gas_features(T["long"], verbose=verbose)
    features = features.join(avg_price, how="left").join(avg_limit, how="left")

    # Step 6: Align to target address list + labels (no leakage)
    if verbose:
        util.print_heading("Aligning Features to Target Address List")

    target_index = pd.Index(target_addresses)
    features = features.reindex(index=target_index)

    labels_idx = addr_labels.set_index("Address")["Scam"]
    features["Scam"] = labels_idx.loc[features.index].astype(int)

    # Step 7: Final fill strategy (counts as Int64, everything else as float)

    # A. Integer-like columns (counts & degrees)
    degree_cols = [
        "in_degree",
        "out_degree",
        "unique_in_degree",
        "unique_out_degree",
        "Incoming count",
        "Outgoing count",
    ]
    for c in degree_cols:
        if c not in features.columns:
            features[c] = 0
    features[degree_cols] = features[degree_cols].fillna(0).astype("Int64")

    # B. Amount columns (floats)
    amount_cols = [
        "Avg amount incoming",
        "Total amount incoming",
        "Max amount incoming",
        "Min amount incoming",
        "Avg amount outgoing",
        "Total amount outgoing",
        "Max amount outgoing",
        "Min amount outgoing",
    ]
    for c in amount_cols:
        if c not in features.columns:
            features[c] = 0.0
    features[amount_cols] = features[amount_cols].fillna(0.0)

    # C. All other numeric columns (time, gas, ratios, entropy, etc.)
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    extra_cols = [c for c in num_cols if c not in degree_cols + amount_cols + ["Scam"]]
    features[extra_cols] = features[extra_cols].fillna(0.0)

    # Lightweight preview + summary only when verbose=True
    if verbose:
        util.print_sub_heading(f"Final Feature Table Sample — {split_label}")
        display(features.head())
        print("Total addresses in this split:", len(features))
        print("Total scam labels:", int(features["Scam"].sum()))

    return features