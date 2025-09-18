import pandas as pd
import duckdb
import os
import glob

# Paths
csv_folder = r"C:\Users\peppe\OneDrive\Desktop\Charlie\Data_Projects\tennis-point-by-point\outputs\all_points_with_importance.csv"
merged_csv = r"C:\Users\peppe\OneDrive\Desktop\Charlie\Data_Projects\tennis-point-by-point\data\processed\merged_tennis_data.csv"
duckdb_file = r"C:\Users\peppe\OneDrive\Desktop\Charlie\Data_Projects\tennis-point-by-point\outputs\sim_results.duckdb"
table_name = "importance_results"

# --- Load all Spark CSV parts ---
csv_files = glob.glob(os.path.join(csv_folder, "part-*.csv"))
spark_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# --- Load updated match winners ---
merged_df = pd.read_csv(merged_csv)
match_winner_df = merged_df[['match_id', 'match_winner']].drop_duplicates()

# --- Merge match winners ---
spark_df = spark_df.drop(columns=['match_winner'], errors='ignore')
spark_df = spark_df.merge(match_winner_df, on='match_id', how='left')

# --- Overwrite original CSV ---
output_csv = os.path.join(csv_folder, "all_points_with_importance.csv")
spark_df.to_csv(output_csv, index=False)
print(f"CSV updated in place: {output_csv}")

# --- Update DuckDB ---
con = duckdb.connect(duckdb_file)
con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM spark_df")
con.close()
print(f"DuckDB table '{table_name}' updated in {duckdb_file}")
