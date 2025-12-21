"""Filter out rows without Skills from candidates CSV."""
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
original_path = data_dir / "candidates_dataset.csv"
clean_path = data_dir / "candidates_dataset_clean.csv"

# Read original file
df = pd.read_csv(original_path)
print(f"Before filtering: {len(df)} rows")

# Check Skills
null_skills = df['Skills'].isna().sum()
empty_skills = (df['Skills'] == '').sum()
whitespace_skills = (df['Skills'].str.strip() == '').sum() if not df['Skills'].isna().all() else 0
print(f"Null Skills: {null_skills}")
print(f"Empty Skills: {empty_skills}")
print(f"Whitespace-only Skills: {whitespace_skills}")

# Filter: keep only rows with non-null and non-empty Target AND Skills
df_filtered = df[
    df['Target'].notna() & (df['Target'].str.strip() != '') &
    df['Skills'].notna() & (df['Skills'].str.strip() != '')
]
print(f"After filtering: {len(df_filtered)} rows")
print(f"Removed: {len(df) - len(df_filtered)} rows")

# Save
df_filtered.to_csv(clean_path, index=False)
print(f"Saved to {clean_path}")
