import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config

# Load the CSV file containing centre, partno, and group
csv_path = config.CASE_CONTROL_RAW
df_csv = pd.read_csv(csv_path)

# Create FemNAT-ID formatted as "XX-YYYY"
df_csv["ID_FemNAT"] = df_csv["centre"].apply(lambda x: f"{int(x):02d}") + "-" + df_csv["partno"].apply(lambda x: f"{int(x):04d}")

# Keep only FemNAT-ID and group columns
df_csv = df_csv[["ID_FemNAT", "group"]]

# Load the Excel file with ID_LifeAndBrain and ID_FemNAT
excel_path = config.ID_EXCEL
df_excel = pd.read_excel(excel_path)

# Merge to attach group info to each ID_LifeAndBrain
merged_df = df_excel.merge(df_csv, on="ID_FemNAT", how="left")

# Final output: ID_LifeAndBrain → group
result_df = merged_df[["ID_LifeAndBrain", "group"]]

# Drop rows where either column is missing or empty
result_df = result_df.dropna(subset=["ID_LifeAndBrain", "group"])
result_df = result_df[(result_df["ID_LifeAndBrain"].astype(str).str.strip() != "") &
                      (result_df["group"].astype(str).str.strip() != "")]

# Save cleaned data as CSV
result_df.to_csv("/files/data/Final_Imputed/match_ids/ID_LifeAndBrain_to_Group.csv", index=False)
