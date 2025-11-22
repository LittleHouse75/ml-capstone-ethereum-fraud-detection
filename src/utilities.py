import pandas as pd

SECTION_DIVIDER = '=' * 40
SUB_SECTION_DIVIDER = '-' * 40
DATA_PATH = 'data/Dataset.csv'

def print_heading(title: str):
    """Print a visually distinct major section header."""
    print(f"\n{SECTION_DIVIDER}\n{title}\n{SECTION_DIVIDER}\n")

def print_sub_heading(title: str):
    """Print a visually distinct sub-section header."""
    print(f"\n{SUB_SECTION_DIVIDER}\n{title}\n{SUB_SECTION_DIVIDER}\n")

def load_raw_data():
    print_heading('Loading Raw Dataset')
    print(f'Reading dataset from: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)
    print(f'Dataset loaded successfully with {len(df):,} rows and {len(df.columns)} columns.')
    return df

