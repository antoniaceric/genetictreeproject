# Lookup what subject number which ID has

from pathlib import Path
import config
import sys

# Load IIDs from sample_ids.txt
id_path = config.SAMPLE_IDS_TXT
with open(id_path, "r") as f:
    sample_ids = [line.strip() for line in f.readlines()]

# Batch mode: CLI arguments
if len(sys.argv) > 1:
    iids = sys.argv[1:]
    print("Batch mode: looking up", len(iids), "IIDs from command line.")

    for iid in iids:
        if iid in sample_ids:
            idx = sample_ids.index(iid)
            img_name = f"subject_{idx:03}_sample_0000.png"
            print(f"{iid} → index = {idx}, image file = {img_name}")
        else:
            print(f"{iid} → NOT FOUND")
    sys.exit()

# Interactive mode
print("Loaded", len(sample_ids), "sample IDs.")
print("Interactive mode: enter one or more IIDs (space or comma separated), or type 'exit' to quit.")

while True:
    iids_input = input("\nEnter IID(s): ").strip()

    if iids_input.lower() in ("exit", "quit"):
        print("Exiting.")
        break

    # Support comma or space-separated input
    iids = [x.strip() for x in iids_input.replace(",", " ").split()]
    
    for iid in iids:
        if iid in sample_ids:
            idx = sample_ids.index(iid)
            img_name = f"subject_{idx:03}_sample_0000.png"
            print(f"{iid} → index = {idx}, image file = {img_name}")
        else:
            print(f"{iid} → NOT FOUND")
