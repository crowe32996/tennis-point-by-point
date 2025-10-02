import pandas as pd
from pathlib import Path
import json

# repo root (assuming this script is in scripts/)
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "player_countries.csv"
PLAYER_MAPPING_FILE = BASE_DIR / "outputs" / "player_mapping.json"

YEARS = [2020, 2021, 2022, 2023, 2024]

BASE_URLS = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"
}

player_country_map = {}

# Load canonical player mapping
with open(PLAYER_MAPPING_FILE, "r") as f:
    player_mapping = json.load(f)  # {variant_name: canonical_name, ...}

def to_initial_format(name):
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return name

def canonical_name(name):
    """Return the canonical name if it exists in mapping, else the original."""
    return player_mapping.get(name, name)

def build_player_country_map(tour="atp"):
    base_url = BASE_URLS[tour]
    for year in YEARS:
        url = base_url.format(year=year)
        df = pd.read_csv(url)
        
        for _, row in df.iterrows():
            winner = canonical_name(to_initial_format(row['winner_name']))
            loser = canonical_name(to_initial_format(row['loser_name']))
            player_country_map[winner] = row['winner_ioc']
            player_country_map[loser] = row['loser_ioc']

# Build ATP + WTA mapping
build_player_country_map("atp")
build_player_country_map("wta")

# Convert to DataFrame and drop duplicates
player_country_df = pd.DataFrame(
    list(player_country_map.items()), columns=['player', 'country']
).drop_duplicates()

player_country_df.to_csv(
    OUTPUT_FILE,
    index=False
)
print("Player-country mapping saved to processed/player_countries.csv")