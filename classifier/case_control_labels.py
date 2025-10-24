import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config

# Load mapping from CSV (1 = control, 2 = case)
csv_path = config.CASE_CONTROL_CSV
df = pd.read_csv(csv_path)
label_dict = dict(zip(df["ID_LifeAndBrain"], df["group"]))

# Load ordered sample IDs
with open(config.SAMPLE_IDS_TXT, "r") as f:
    sample_ids = [line.strip() for line in f]

print(f"Loaded {len(sample_ids)} sample IDs.")
labels = []
missing = []
matched_data = []

# Match each sample ID to a group
for idx, iid in enumerate(sample_ids):
    group = label_dict.get(iid, -1)  # -1 for missing
    labels.append(group)
    matched_data.append((idx, iid, group))
    if group == -1:
        missing.append(iid)

labels = np.array(labels)

# Warn about missing
if missing:
    print(f"{len(missing)} sample IDs have no label in CSV.")
    print("Example missing IDs:", missing[:5])
    # Optional strict mode:
    # raise ValueError("Missing labels for some sample IDs.")

# Save full CSV with subject index
df_out = pd.DataFrame(matched_data, columns=["subject_index", "ID_LifeAndBrain", "group"])
csv_output = config.EMBEDDING_DIR / "case_control_labels.csv"
df_out.to_csv(csv_output, index=False)
print(f"Saved matched label CSV to: {csv_output}")
