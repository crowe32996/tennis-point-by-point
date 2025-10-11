import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PLAYER_COUNTRY_FILE = BASE_DIR / "data" / "processed" / "player_countries.csv"


def add_basic_columns(df):
    # --- win probability deltas (vectorized, faster than apply) ---
    if "p1_wp_delta" not in df.columns and all(col in df.columns for col in ["PointWinner", "p1_win_prob_if_p1_wins", "p1_win_prob_if_p2_wins", "p1_win_prob_before"]):
        df["p1_wp_delta"] = np.where(
            df["PointWinner"] == 1,
            df["p1_win_prob_if_p1_wins"] - df["p1_win_prob_before"],
            df["p1_win_prob_if_p2_wins"] - df["p1_win_prob_before"]
        )
        df["p2_wp_delta"] = -df["p1_wp_delta"]

    # --- player name cleanup ---
    if "player1" in df.columns and "player2" in df.columns:
        df["player1"] = df["player1"].map(clean_player_name)
        df["player2"] = df["player2"].map(clean_player_name)

    # --- gender + Tour extraction ---
    # if "match_id" in df.columns:
    #     # --- Tournament extraction from match_id ---
    #     # Example match_id: 2023-ausopen-1120
    #     df["tourney_code"] = df["match_id"].astype(str).str.split("-", n=2).str[1]
    #     df["tournament"] = df["tourney_code"].map(TOURNAMENTS_MAP).fillna(df["tourney_code"])

    return df

def filter_matches_by_sets(df: pd.DataFrame) -> pd.DataFrame:
    """S
    Remove invalid matches from the dataset:
      - Men: exclude if a player has 3 sets won or more
      - Women: exclude if a player has 2 sets won or more
    """
    def is_valid(group):
        tour = group["tour"].iloc[0]
        p1_sets = group["p1_sets_won"].max()
        p2_sets = group["p2_sets_won"].max()
        if tour in ("ATP",'All') and (p1_sets >= 3 or p2_sets >= 3):
            return False
        if tour == "WTA" and (p1_sets >= 2 or p2_sets >= 2):
            return False
        return True

    return df.groupby("match_id", group_keys=False).filter(is_valid).reset_index(drop=True)


def apply_filters(df, selected_tourney, selected_tour, selected_years):
    df2 = df
    df2["tournament"] = df2["tournament"].map(TOURNAMENTS_MAP).fillna(df2["tournament"])
    if selected_tourney != "All":
        df2 = df2[df2["tournament"] == selected_tourney]
    if selected_tour != "All":
        df2 = df2[df2["Tour"] == selected_tour]
    if selected_years:
        df2 = df2[df2["year"].isin(selected_years)]
    return df2

def add_filtered_player_columns(df, selected_players):
    most_recent_year = df['year'].max()
    active_players = pd.unique(
        df[df['year'] == most_recent_year][['player1', 'player2']].values.ravel()
    )
    inactive_players = [p for p in pd.unique(df[['player1', 'player2']].values.ravel())
                        if p not in active_players]

    def mask_player(player_name):
        if selected_players == "All":
            return player_name
        elif selected_players == "Active":
            return player_name if player_name in active_players else None
        elif selected_players == "Inactive":
            return player_name if player_name in inactive_players else None

    df['player1'] = df['player1'].apply(mask_player)
    df['player2'] = df['player2'].apply(mask_player)
    return df


IOC_TO_ISO2 = {
    "SUI": "CH",
    "DEN": "DK",
    "GER": "DE",
    "BLR": "BY",
    "POL": "PL",
    "TUN": "TN",
    "UKR": "UA",
    "BUL": "BG",
    "SRB": "RS",
    "KAZ": "KZ",
    "RSA": "ZA",
    "RUS": "RU",
    "TPE": "TW",
    "PUR": "PR",
    "ESA": "SV",
    "INA": "ID",
    "VAN": "VU",
    "NMI": "MP",
    "POC": "XK",
    "IRI": "IR",
    "SWE": "SE",  
    "CHI": "CL",  
    "AUT": "AT",  
    "NED": "NL",
    "CRO": "HR",
    "KOR":"KR",
    "POR": "PT",
    "CHN": "CN",  
}

def ioc_to_flag(ioc_code):
    iso2 = IOC_TO_ISO2.get(ioc_code, ioc_code[:2].upper())
    OFFSET = 127397
    return "".join([chr(ord(c) + OFFSET) for c in iso2])

def load_player_flag_map(file_path=PLAYER_COUNTRY_FILE):
    """Return a dict mapping players to their country/flag."""
    df = pd.read_csv(file_path)
    return dict(zip(df["player"], df["country"]))
    
def add_flag_to_player(df, player_flag_map, player_col="Player"):
    df[player_col] = df[player_col].map(
        lambda x: f"{ioc_to_flag(player_flag_map.get(x, ''))} {x}" if player_flag_map.get(x) else x
    )
    return df

def add_flag_image(df, player_flag_map, player_col="Player"):
    def flag_img_html(player_name):
        ioc = player_flag_map.get(player_name, "")
        if ioc:
            iso2 = IOC_TO_ISO2.get(ioc, ioc[:2].upper())
            url = f"https://flagcdn.com/w20/{iso2.lower()}.png"
            return f'<img src="{url}" width="20" style="vertical-align:middle;margin-right:4px">{player_name}'
        else:
            return player_name

    df[player_col] = df[player_col].apply(flag_img_html)
    return df


def render_flag_table(df, player_flag_map, player_col="Player", numeric_cols=None, max_height=400):
    """
    Render a DataFrame with flags using HTML and a scrollable container.
    df: dataframe with player names and stats
    player_col: column with player names
    numeric_cols: list of numeric/stat columns to display
    max_height: max height of table in pixels (scrollbar appears if exceeded)
    """
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if c != player_col]

    html = '<div style="overflow-y:visible;">'
    html += '<table style="width:100%; border-collapse: collapse;">'
    html += "<tr><th style='text-align:left'>Player</th>"
    for col in numeric_cols:
        html += f"<th style='text-align:right'>{col}</th>"
    html += "</tr>"

    for _, row in df.iterrows():
        player_name = row[player_col]
        ioc_code = player_flag_map.get(player_name, "")
        flag_html = ""
        if ioc_code:
            iso2 = IOC_TO_ISO2.get(ioc_code, ioc_code[:2].upper())
            url = f"https://flagcdn.com/w20/{iso2.lower()}.png"
            flag_html = f'<img src="{url}" width="20" style="vertical-align:middle;margin-right:4px">'
        html += "<tr>"
        html += f"<td>{flag_html}{player_name}</td>"
        for col in numeric_cols:
            val = row[col]
            if isinstance(val, float):
                if col in ["High Pressure %","Win % (High Pressure)"]:
                    html += f"<td style='text-align:right'>{val*100:.1f}%</td>"
                else:
                    html += f"<td style='text-align:right'>{val:.3f}</td>"
            else:
                html += f"<td style='text-align:right'>{val}</td>"
        html += "</tr>"
    html += "</table></div>"
    return html


def render_scoreboard(row, height = 130):
    tournament_logo = "🎾"

    # Add serve emoji to serving player
    p1_name = f"{row['Player 1']}"
    p2_name = f"{row['Player 2']}"
    if str(row["Server"]) in ["1", "Player 1"]:
        p1_name += " 🎾"
        server_first = True
    else:
        p2_name += " 🎾"
        server_first = False

    # Bold winner
    if row["Match Winner"] == row["Player 1"]:
        p1_name = f"<strong>{p1_name}</strong>"
    else:
        p2_name = f"<strong>{p2_name}</strong>"

    if "-" in str(row["Game Score"]):
        score_parts = row["Game Score"].split("-", 1)
        score_p1 = score_parts[0]
        score_p2 = score_parts[1]
    else:
        score_p1 = row["Game Score"]
        score_p2 = row["Game Score"]

    html = f"""
    <div style="
        border: 2px solid #ddd; 
        border-radius: 0px; 
        padding: 2px; 
        margin-bottom: 0px; 
        box-shadow: 1px 1px 4px rgba(0,0,0,0.08);
        font-family: Arial, sans-serif;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0px;">
            <span style="font-weight: bold; font-size: 1em;">{row['Tournament']} {row['Year']}</span>
            <span style="font-size: 1.2em;">{tournament_logo}</span>
        </div>

        <table style="width:100%; text-align: center; border-collapse: collapse; font-size: 0.9em;">
            <tr>
                <th style="text-align:left;">Player</th>
                <th>Sets</th>
                <th>Games</th>
                <th>Game Score</th>
            </tr>
            <tr>
                <td style="text-align:left;">{p1_name}</td>
                <td>{row['P1 Sets']}</td>
                <td>{row['P1 Games']}</td>
                <td>{score_p1}</td>
            </tr>
            <tr>
                <td style="text-align:left;">{p2_name}</td>
                <td>{row['P2 Sets']}</td>
                <td>{row['P2 Games']}</td>
                <td>{score_p2}</td>
            </tr>
        </table>

        <div style="margin-top:2px; font-size:0.85em; color:#555; text-align:center;">
            Lowest Win Probability: {row['Win Probability']:.1f}%
        </div>
    </div>
    """

    # Render the HTML inside an iframe (set height so iframe matches content)
    return html

def make_score_heatmap(df):
    return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_rect()