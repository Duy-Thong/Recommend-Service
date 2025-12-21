import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("c:/Users/DaoDuyThong/StudioProjects/Recommend Service/evaluation_data/ground_truth.csv")

bins = [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
labels = ["< 0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]

for (low, high), label in zip(bins, labels):
    subset = df[(df["similarity"] >= low) & (df["similarity"] < high)]
    print(f"\n{'='*60}")
    print(f"{label} ({len(subset)} records)")
    print("="*60)
    samples = subset.head(2)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n  Sample {i}:")
        print(f"    CV:  {row['cv_title']}")
        print(f"    Job: {row['job_title']}")
        print(f"    Similarity: {row['similarity']:.4f}")
