import pandas as pd
from pathlib import Path

# repo root (assuming this script is in scripts/)
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "player_countries.csv"

YEARS = [2020, 2021, 2022, 2023, 2024]

BASE_URLS = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"
}

player_country_map = {}

def to_initial_format(name):
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return name

def build_player_country_map(tour="atp"):
    base_url = BASE_URLS[tour]
    for year in YEARS:
        url = base_url.format(year=year)
        df = pd.read_csv(url)
        
        for _, row in df.iterrows():
            winner = to_initial_format(row['winner_name'])
            loser = to_initial_format(row['loser_name'])
            player_country_map[winner] = row['winner_ioc']
            player_country_map[loser] = row['loser_ioc']

# Example usage: build ATP + WTA mapping
build_player_country_map("atp")
build_player_country_map("wta")

# hard-coded player name exceptions
player_country_map["A. Ramos Vinolas"] = player_country_map.pop("A. Ramos", "ESP")
player_country_map["C. O'Connell"] = player_country_map.pop("C. Oconnell", "IRL")
player_country_map["F. Auger-Aliassime"] = player_country_map.pop("F. Auger Aliassime", "CAN")

# Convert to DataFrame
player_country_df = pd.DataFrame(
    list(player_country_map.items()), columns=['player', 'country']
).drop_duplicates()

player_country_df.to_csv(
    OUTPUT_FILE,
    index=False
)
print("Player-country mapping saved to processed/player_countries.csv")
