import duckdb
import pandas as pd
from pathlib import Path
from transform import transform_tables

# Base directory is the repo root
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed
if not (BASE_DIR / "data").exists():
    BASE_DIR = Path(__file__).resolve().parent  # fallback for cloud

DUCKDB_FILE = BASE_DIR / "outputs" / "sim_results.duckdb"
# OUTPUT_FILE = BASE_DIR / "outputs" / "table_preview.xlsx"

# Connect to DuckDB
con = duckdb.connect(DUCKDB_FILE)

# Pull player info
df_players = con.execute("""
    SELECT *
    FROM player_detail
""").fetchdf()

con.close()

df_players.to_csv("player_detail_export.csv", index=False)

