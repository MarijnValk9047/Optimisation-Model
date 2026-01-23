import pandas as pd

A = pd.read_csv(r"/1_Configs/1.2_Data_processed/DA-market/03_regimes/strategy_A/A_representation_error_by_split.csv")
B = pd.read_csv(r"/1_Configs/1.2_Data_processed/DA-market/03_regimes/strategy_B/B_representation_error_by_split.csv")

m = A.merge(B, on=["class_id","k","split"], suffixes=("_A","_B"))
for col in ["mean_dist","p90_dist","p95_dist","max_dist"]:
    m[f"delta_{col}"] = m[f"{col}_B"] - m[f"{col}_A"]

print(m.sort_values(["split","class_id"])[
    ["class_id","split","k",
     "mean_dist_A","mean_dist_B","delta_mean_dist",
     "p95_dist_A","p95_dist_B","delta_p95_dist"]
].to_string(index=False))
