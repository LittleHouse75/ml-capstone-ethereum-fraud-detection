#!/usr/bin/env python

import os
import time
from pathlib import Path

import requests
import pandas as pd

print(">>> fetch_dfpi_txs.py starting up")

# -----------------------------------
# Config / paths
# -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
print("PROJECT_ROOT:", PROJECT_ROOT)

ADDRESSES_FILE = PROJECT_ROOT / "data" / "external" / "dfpi_eth_nonscam.txt"
RAW_DIR = PROJECT_ROOT / "data" / "external" / "dfpi_raw"
COMBINED_CSV = PROJECT_ROOT / "data" / "external" / "dfpi_all_txs.csv"

print("ADDRESSES_FILE:", ADDRESSES_FILE)
print("RAW_DIR:", RAW_DIR)
print("COMBINED_CSV:", COMBINED_CSV)

RAW_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("ETHERSCAN_API_KEY")
print("ETHERSCAN_API_KEY present?", bool(API_KEY))

if not API_KEY:
    raise RuntimeError("ETHERSCAN_API_KEY not set in environment")


# -----------------------------------
# Helper: load scam addresses
# -----------------------------------
def load_addresses() -> list[str]:
    print("Loading addresses from:", ADDRESSES_FILE)
    if not ADDRESSES_FILE.exists():
        raise FileNotFoundError(f"Addresses file not found: {ADDRESSES_FILE}")
    with ADDRESSES_FILE.open() as f:
        addrs = [line.strip() for line in f if line.strip()]
    seen = set()
    uniq = []
    for a in addrs:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    print(f"Loaded {len(uniq)} unique addresses")
    return uniq


# -----------------------------------
# Helper: fetch transactions for one address
# -----------------------------------
def fetch_txs_for_address(address: str) -> list[dict]:
    url = "https://api.etherscan.io/v2/api"
    params = {
        "apikey": API_KEY,
        "chainid": "1",          # Ethereum mainnet
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 9999999999,
        "page": 1,
        "offset": 10000,         # max per docs
        "sort": "asc",
    }
    print(f"  [HTTP] GET {url} for {address}")
    resp = requests.get(url, params=params, timeout=30)
    print(f"  [HTTP] status_code={resp.status_code}")
    resp.raise_for_status()

    data = resp.json()
    status = data.get("status")
    message = data.get("message")
    result = data.get("result")
    print(f"  [HTTP] Etherscan status={status}, message={message}")

    if status != "1":
        print(f"  [WARN] {address}: status={status}, message={message}, result={result}")
        return []

    return result


# -----------------------------------
# Main
# -----------------------------------
def main():
    print(">>> main() entered")
    addresses = load_addresses()
    print(f"Will process {len(addresses)} addresses")

    all_rows = []

    for i, addr in enumerate(addresses, start=1):
        out_path = RAW_DIR / f"{addr}.csv"
        print(f"[{i}/{len(addresses)}] Address {addr}")

        if out_path.exists():
            print(f"  Already fetched, loading {out_path}")
            df = pd.read_csv(out_path)
            all_rows.append(df)
            continue

        try:
            txs = fetch_txs_for_address(addr)
        except Exception as e:
            print(f"  ERROR fetching {addr}: {e}")
            continue

        if not txs:
            print(f"  No transactions returned for {addr}")
            continue

        df = pd.DataFrame(txs)
        df["dfpi_scam_address"] = addr

        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} txs to {out_path}")

        all_rows.append(df)

        time.sleep(0.2)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(COMBINED_CSV, index=False)
        print(f">>> Combined {len(combined)} tx rows -> {COMBINED_CSV}")
    else:
        print(">>> No transactions fetched; combined file not written.")


if __name__ == "__main__":
    print(">>> __name__ == '__main__', calling main()")
    main()
else:
    print(">>> fetch_dfpi_txs.py imported as a module (not run as script)")