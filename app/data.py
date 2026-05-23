import pandas as pd
import duckdb
import streamlit as st
from pathlib import Path
from transform import transform_tables

# Base directory is the repo root
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed
if not (BASE_DIR / "data").exists():
    BASE_DIR = Path(__file__).resolve().parent  # fallback for cloud

DUCKDB_FILE = BASE_DIR / "outputs" / "sim_results.duckdb"
TABLE_NAME = "importance_results"
PLAYER_COUNTRY_FILE = BASE_DIR / "data" / "processed" / "player_countries.csv"

# Initialize tables on module load
transform_tables()

player_country_df = pd.read_csv(PLAYER_COUNTRY_FILE)
player_flag_map = dict(zip(player_country_df["player"], player_country_df["country"]))


def clean_player_name(name: str) -> str:
    """Capitalize player names correctly, handling spaces, hyphens, and apostrophes."""
    if not name or not name.strip():
        return name

    parts = name.split()
    first, rest_parts = parts[0], parts[1:]

    cleaned_rest = []
    for word in rest_parts:
        for sep in ["-", "'"]:
            if sep in word:
                word = sep.join([w.capitalize() for w in word.split(sep)])
        cleaned_rest.append(word.capitalize() if all(sep not in word for sep in ["-", "'"]) else word)

    return f"{first} {' '.join(cleaned_rest)}".strip()

@st.cache_data
def load_tab0_sql(selected_years, selected_tour, selected_tourney,
                  selected_players, min_points_filter=None):
    """
    Load DuckDB data for Tab 0 with all filters applied.
    """
    years_str = ",".join(map(str, selected_years))
    con = duckdb.connect(DUCKDB_FILE)

    query = f"""
    SELECT
        pt.match_id,
        m.player1,
        m.player2,
        m.year,
        pt.point_winner,
        pt.point_server,
        pt.server_name,
        pt.returner_name,
        pp.p1_win_prob_before,
        pp.p1_win_prob_if_p1_wins,
        pp.p1_win_prob_if_p2_wins,
        pp.p1_wp_delta,
        pp.p2_wp_delta,
        pp.importance,
        m.points_stake
    FROM point_detail pt
    JOIN match_detail m USING(match_id)
    JOIN point_probability pp USING(match_id, point_number)
    JOIN player_detail pd1 ON pd1.player = m.player1
    JOIN player_detail pd2 ON pd2.player = m.player2
    WHERE m.year IN ({years_str})
      AND ('{selected_tour}' = 'All' OR m.tour = '{selected_tour}')
      AND ('{selected_tourney}' = 'All' OR m.tournament_name = '{selected_tourney}')
      AND ('{selected_players}' = 'All'
           OR pd1.player_status = '{selected_players}'
           OR pd2.player_status = '{selected_players}')
    """

    df = con.execute(query).fetchdf()
    con.close()
    return df


@st.cache_data
def load_tab1_sql(selected_years, selected_tour, selected_tourney,
                  selected_players, min_points_filter=None):
    """
    Load DuckDB data for Tab 0 with all filters applied.
    """
    years_str = ",".join(map(str, selected_years))
    con = duckdb.connect(DUCKDB_FILE)

    query = f"""
    SELECT
        pt.match_id,
        m.player1,
        m.player2,
        m.year,
        m.tour,
        pt.point_winner,
        pt.point_server,
        pt.server_name,
        pt.returner_name,
        pp.p1_win_prob_before,
        pp.p1_win_prob_if_p1_wins,
        pp.p1_win_prob_if_p2_wins,
        pp.p1_wp_delta,
        pp.p2_wp_delta,
        pp.importance,
        m.points_stake
    FROM point_detail pt
    JOIN match_detail m USING(match_id)
    JOIN point_probability pp USING(match_id, point_number)
    JOIN player_detail pd1 ON pd1.player = m.player1
    JOIN player_detail pd2 ON pd2.player = m.player2
    WHERE m.year IN ({years_str})
      AND ('{selected_tour}' = 'All' OR m.tour = '{selected_tour}')
      AND ('{selected_tourney}' = 'All' OR m.tournament_name = '{selected_tourney}')
      AND ('{selected_players}' = 'All'
           OR pd1.player_status = '{selected_players}'
           OR pd2.player_status = '{selected_players}')
    """

    df = con.execute(query).fetchdf()
    con.close()
    return df


@st.cache_data
def load_tab2_sql(selected_years, selected_tour, selected_tourney,
                  selected_players, min_points_filter=None):
    """
    Load DuckDB data for Tab 0 with all filters applied.
    """
    years_str = ",".join(map(str, selected_years))
    con = duckdb.connect(DUCKDB_FILE)

    query = f"""
    SELECT
        pt.match_id,
        m.player1,
        m.player2,
        m.year,
        m.tour,
        m.tournament_name,
        m.match_winner,
        pt.point_winner,
        pt.point_server,
        pt.server_name,
        pt.returner_name,
        pt.p1_sets_won,
        pt.p2_sets_won,
        pt.p1_games_won,
        pt.p2_games_won,
        pt.score,
        pt.point_number,
        pp.p1_win_prob_before,
        pp.p1_win_prob_if_p1_wins,
        pp.p1_win_prob_if_p2_wins,
        pp.p1_wp_delta,
        pp.p2_wp_delta,
        pp.importance,
        m.points_stake
    FROM point_detail pt
    JOIN match_detail m USING(match_id)
    JOIN point_probability pp USING(match_id, point_number)
    JOIN player_detail pd1 ON pd1.player = m.player1
    JOIN player_detail pd2 ON pd2.player = m.player2
    WHERE m.year IN ({years_str})
      AND ('{selected_tour}' = 'All' OR m.tour = '{selected_tour}')
      AND ('{selected_tourney}' = 'All' OR m.tournament_name = '{selected_tourney}')
      AND ('{selected_players}' = 'All'
           OR pd1.player_status = '{selected_players}'
           OR pd2.player_status = '{selected_players}')
    """

    df = con.execute(query).fetchdf()
    con.close()
    return df