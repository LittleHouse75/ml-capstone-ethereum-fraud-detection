import pandas as pd

def normalize_timestamps(
    df: pd.DataFrame,
    ts_col: str = "block_timestamp",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Normalize raw timestamp strings into a clean UTC datetime and add
    derived time-of-day / weekday features.

    Mutates df in-place and also returns it.
    Adds:
      - f"{ts_col}_raw"  (original as string)
      - f"{ts_col}_std"  (cleaned string)
      - f"{ts_col}_dt"   (parsed datetime, UTC)
      - 'hour'           (hour-of-day)
      - 'weekday'        (day name)
    """

    print_sub_heading("Purpose")
    print(
        "Convert raw timestamps into a consistent UTC datetime. "
        "Raw strings are preserved in `block_timestamp`."
    )
        
    if verbose:
        print("types:\n", df[ts_col].apply(type).value_counts())
        print("isna:\n", df[ts_col].isna().sum())

    raw_col = f"{ts_col}_raw"
    std_col = f"{ts_col}_std"
    dt_col  = f"{ts_col}_dt"

    # 1. Preserve original raw text for audit/debugging
    df[raw_col] = df[ts_col].astype(str)

    # 2. Normalize unicode and convert to clean working string
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

    # 5. Derive features
    df["hour"] = df[dt_col].dt.hour
    df["weekday"] = df[dt_col].dt.day_name()

    if verbose:
        print("Parsed timestamps:", df[dt_col].notna().sum())
        print("Unparseable timestamps:", df[dt_col].isna().sum())

    return df