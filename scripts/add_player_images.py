"""
Fetch player headshots from TheSportsDB (free API).
Stores results in player_images.csv
"""
import requests
import pandas as pd
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent
PLAYER_COUNTRY_FILE = BASE_DIR / "data" / "processed" / "player_countries.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "player_images.csv"

# Load existing players
players_df = pd.read_csv(PLAYER_COUNTRY_FILE)


def expand_name(short_name: str) -> str:
    """Convert 'N. Djokovic' to 'Novak Djokovic' style for better search."""
    # Common first name expansions for top players
    expansions = {
        "N. Djokovic": "Novak Djokovic",
        "R. Nadal": "Rafael Nadal",
        "R. Federer": "Roger Federer",
        "C. Alcaraz": "Carlos Alcaraz",
        "D. Medvedev": "Daniil Medvedev",
        "A. Zverev": "Alexander Zverev",
        "S. Tsitsipas": "Stefanos Tsitsipas",
        "J. Sinner": "Jannik Sinner",
        "A. Rublev": "Andrey Rublev",
        "H. Rune": "Holger Rune",
        "F. Auger-Aliassime": "Felix Auger-Aliassime",
        "T. Fritz": "Taylor Fritz",
        "C. Ruud": "Casper Ruud",
        "H. Hurkacz": "Hubert Hurkacz",
        "N. Kyrgios": "Nick Kyrgios",
        "A. De Minaur": "Alex De Minaur",
        "D. Shapovalov": "Denis Shapovalov",
        "M. Berrettini": "Matteo Berrettini",
        "G. Monfils": "Gael Monfils",
        "S. Wawrinka": "Stan Wawrinka",
        "K. Khachanov": "Karen Khachanov",
        "B. Shelton": "Ben Shelton",
        "F. Tiafoe": "Frances Tiafoe",
        "A. Murray": "Andy Murray",
        "G. Dimitrov": "Grigor Dimitrov",
        # WTA
        "I. Swiatek": "Iga Swiatek",
        "A. Sabalenka": "Aryna Sabalenka",
        "C. Gauff": "Coco Gauff",
        "E. Rybakina": "Elena Rybakina",
        "J. Pegula": "Jessica Pegula",
        "O. Jabeur": "Ons Jabeur",
        "M. Sakkari": "Maria Sakkari",
        "P. Badosa": "Paula Badosa",
        "B. Haddad Maia": "Beatriz Haddad Maia",
        "M. Keys": "Madison Keys",
        "V. Azarenka": "Victoria Azarenka",
        "P. Kvitova": "Petra Kvitova",
        "S. Williams": "Serena Williams",
        "N. Osaka": "Naomi Osaka",
        "S. Halep": "Simona Halep",
        "K. Pliskova": "Karolina Pliskova",
        "D. Collins": "Danielle Collins",
        "E. Svitolina": "Elina Svitolina",
        "B. Andreescu": "Bianca Andreescu",
        "E. Raducanu": "Emma Raducanu",
    }
    return expansions.get(short_name, short_name)


def search_player(name: str) -> dict | None:
    """Search TheSportsDB for player headshot."""
    # Try expanded name first
    search_name = expand_name(name)

    # Also try just the last name if initial format
    if ". " in name:
        last_name = name.split(". ", 1)[1]
    else:
        last_name = name.split()[-1] if name else name

    for query in [search_name, last_name]:
        try:
            url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={query.replace(' ', '%20')}"
            resp = requests.get(url, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                if data.get("player"):
                    # Find tennis player
                    for p in data["player"]:
                        if p.get("strSport") == "Tennis":
                            return {
                                "player": name,
                                "full_name": p.get("strPlayer"),
                                "headshot_url": p.get("strCutout") or p.get("strThumb"),
                                "thumb_url": p.get("strThumb"),
                            }
        except Exception as e:
            print(f"  Error: {e}")

    return None


def main():
    results = []
    found = 0

    # Process in batches with progress
    total = len(players_df)

    for idx, row in players_df.iterrows():
        name = row["player"]

        if idx % 20 == 0:
            print(f"[{idx}/{total}] Processing... ({found} found so far)")

        result = search_player(name)

        if result and result.get("headshot_url"):
            results.append(result)
            found += 1
        else:
            results.append({"player": name, "headshot_url": None})

        # Rate limit - be nice to free API
        time.sleep(0.3)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*50}")
    print(f"Done! Processed {total} players")
    print(f"Found headshots for {found} players ({found/total*100:.1f}%)")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
