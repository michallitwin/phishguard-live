import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_dataset import build_dataset, save_dataset, OUTPUT_PATH

print("Building dataset...")
dataset = build_dataset()
print(f"Built {len(dataset)} rows")
print(dataset["label"].value_counts())

save_dataset(dataset)
print(f"Saved to {OUTPUT_PATH}")
