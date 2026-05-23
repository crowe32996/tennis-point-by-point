"""
FastAPI backend for Tennis Point-by-Point Analysis
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import duckdb
import pandas as pd
from typing import Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import MIN_POINTS_PER_YEAR, HIGH_PRESSURE_PERCENTILE

# Load player country data (file is in api/ folder for deployment)
API_DIR = Path(__file__).resolve().parent
PLAYER_COUNTRY_FILE = API_DIR / "player_countries.csv"
player_country_df = pd.read_csv(PLAYER_COUNTRY_FILE)
PLAYER_COUNTRIES = dict(zip(player_country_df["player"], player_country_df["country"]))

app = FastAPI(
    title="Tennis Pressure Analysis API",
    description="API for analyzing clutch performance and consistency in Grand Slam tennis",
    version="1.0.0"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://tennis.crowedata.com",
        "https://crowedata.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path
BASE_DIR = Path(__file__).resolve().parent.parent
DUCKDB_FILE = BASE_DIR / "outputs" / "sim_results.duckdb"


def get_db():
    """Get database connection."""
    return duckdb.connect(str(DUCKDB_FILE), read_only=True)


def enrich_with_country(df: pd.DataFrame, player_col: str = "player") -> pd.DataFrame:
    """Add country code to dataframe based on player name."""
    df["country"] = df[player_col].map(PLAYER_COUNTRIES).fillna("UNK")
    return df


@app.get("/api/players/countries")
def get_player_countries():
    """Get all player country mappings."""
    return {"players": PLAYER_COUNTRIES}


@app.get("/")
def root():
    return {"message": "Tennis Pressure Analysis API", "docs": "/docs"}


@app.get("/api/filters")
def get_filter_options():
    """Get available filter options for the sidebar."""
    con = get_db()

    tournaments = con.execute(
        "SELECT DISTINCT tournament_name FROM match_detail ORDER BY tournament_name"
    ).fetchall()
    tours = con.execute(
        "SELECT DISTINCT tour FROM match_detail ORDER BY tour"
    ).fetchall()
    years = con.execute(
        "SELECT DISTINCT year FROM match_detail ORDER BY year"
    ).fetchall()

    con.close()

    return {
        "tournaments": ["All"] + [t[0] for t in tournaments],
        "tours": [t[0] for t in tours],
        "years": [y[0] for y in years],
        "player_statuses": ["All", "Active", "Inactive"],
        "defaults": {
            "min_points_per_year": MIN_POINTS_PER_YEAR,
            "high_pressure_percentile": HIGH_PRESSURE_PERCENTILE
        }
    }


@app.get("/api/players/clutch")
def get_player_clutch_rankings(
    years: str = Query(..., description="Comma-separated years, e.g. '2020,2021,2022'"),
    tour: str = Query("ATP", description="ATP or WTA"),
    tournament: str = Query("All", description="Tournament name or 'All'"),
    player_status: str = Query("Active", description="Active, Inactive, or All"),
    min_points: int = Query(400, description="Minimum points played")
):
    """
    Get player clutch rankings (Expected Points Added).
    """
    con = get_db()
    years_list = [int(y.strip()) for y in years.split(",")]
    years_str = ",".join(map(str, years_list))

    query = f"""
    WITH player_points AS (
        SELECT
            CASE WHEN pt.point_winner = 1 THEN m.player1 ELSE m.player2 END as player,
            pp.importance,
            pp.p1_wp_delta * CASE WHEN pt.point_winner = 1 THEN 1 ELSE -1 END as wp_delta,
            m.points_stake,
            m.tour
        FROM point_detail pt
        JOIN match_detail m USING(match_id)
        JOIN point_probability pp USING(match_id, point_number)
        WHERE m.year IN ({years_str})
          AND ('{tour}' = 'All' OR m.tour = '{tour}')
          AND ('{tournament}' = 'All' OR m.tournament_name = '{tournament}')
    )
    SELECT
        player,
        tour,
        SUM(wp_delta * importance * points_stake) as clutch_score,
        COUNT(*) as total_points,
        AVG(wp_delta) as avg_wp_delta
    FROM player_points
    GROUP BY player, tour
    HAVING COUNT(*) >= {min_points}
    ORDER BY clutch_score DESC
    """

    df = con.execute(query).fetchdf()
    con.close()

    # Add country codes
    df = enrich_with_country(df)

    return {
        "players": df.to_dict(orient="records"),
        "top_10": df.head(10).to_dict(orient="records"),
        "bottom_10": df.tail(10).to_dict(orient="records")
    }


@app.get("/api/players/consistency")
def get_player_consistency(
    years: str = Query(..., description="Comma-separated years"),
    tour: str = Query("ATP"),
    tournament: str = Query("All"),
    min_points: int = Query(400)
):
    """
    Get player consistency rankings.
    """
    con = get_db()
    years_list = [int(y.strip()) for y in years.split(",")]
    years_str = ",".join(map(str, years_list))

    query = f"""
    WITH player_deltas AS (
        SELECT
            m.player1 as player,
            pp.p1_wp_delta as wp_delta,
            m.tour
        FROM point_detail pt
        JOIN match_detail m USING(match_id)
        JOIN point_probability pp USING(match_id, point_number)
        WHERE m.year IN ({years_str})
          AND ('{tour}' = 'All' OR m.tour = '{tour}')
          AND ('{tournament}' = 'All' OR m.tournament_name = '{tournament}')

        UNION ALL

        SELECT
            m.player2 as player,
            pp.p2_wp_delta as wp_delta,
            m.tour
        FROM point_detail pt
        JOIN match_detail m USING(match_id)
        JOIN point_probability pp USING(match_id, point_number)
        WHERE m.year IN ({years_str})
          AND ('{tour}' = 'All' OR m.tour = '{tour}')
          AND ('{tournament}' = 'All' OR m.tournament_name = '{tournament}')
    )
    SELECT
        player,
        tour,
        AVG(wp_delta) as avg_delta,
        STDDEV_POP(wp_delta) as delta_std,
        COUNT(*) as total_points,
        CASE WHEN STDDEV_POP(wp_delta) > 0
             THEN AVG(wp_delta) / STDDEV_POP(wp_delta)
             ELSE 0 END as consistency_index
    FROM player_deltas
    GROUP BY player, tour
    HAVING COUNT(*) >= {min_points}
    ORDER BY consistency_index DESC
    """

    df = con.execute(query).fetchdf()
    con.close()

    # Add country codes
    df = enrich_with_country(df)

    return {
        "players": df.to_dict(orient="records"),
        "most_consistent": df.head(10).to_dict(orient="records"),
        "least_consistent": df.tail(10).to_dict(orient="records")
    }


@app.get("/api/players/high-pressure")
def get_high_pressure_stats(
    years: str = Query(...),
    tour: str = Query("ATP"),
    tournament: str = Query("All"),
    pressure_percentile: int = Query(25, description="Top N% of points by importance"),
    min_hp_points: int = Query(50)
):
    """
    Get player performance in high-pressure situations.
    """
    con = get_db()
    years_list = [int(y.strip()) for y in years.split(",")]
    years_str = ",".join(map(str, years_list))

    # First get the importance threshold
    threshold_query = f"""
    SELECT PERCENTILE_CONT({1 - pressure_percentile/100}) WITHIN GROUP (ORDER BY importance) as threshold
    FROM point_probability pp
    JOIN match_detail m USING(match_id)
    WHERE m.year IN ({years_str})
      AND ('{tour}' = 'All' OR m.tour = '{tour}')
    """
    threshold = con.execute(threshold_query).fetchone()[0]

    query = f"""
    WITH player_points AS (
        SELECT
            m.player1 as player,
            pt.point_winner = 1 as won_point,
            pp.importance >= {threshold} as is_high_pressure,
            m.tour
        FROM point_detail pt
        JOIN match_detail m USING(match_id)
        JOIN point_probability pp USING(match_id, point_number)
        WHERE m.year IN ({years_str})
          AND ('{tour}' = 'All' OR m.tour = '{tour}')
          AND ('{tournament}' = 'All' OR m.tournament_name = '{tournament}')

        UNION ALL

        SELECT
            m.player2 as player,
            pt.point_winner = 2 as won_point,
            pp.importance >= {threshold} as is_high_pressure,
            m.tour
        FROM point_detail pt
        JOIN match_detail m USING(match_id)
        JOIN point_probability pp USING(match_id, point_number)
        WHERE m.year IN ({years_str})
          AND ('{tour}' = 'All' OR m.tour = '{tour}')
          AND ('{tournament}' = 'All' OR m.tournament_name = '{tournament}')
    )
    SELECT
        player,
        tour,
        COUNT(*) as total_points,
        SUM(CASE WHEN is_high_pressure THEN 1 ELSE 0 END) as hp_points,
        AVG(CASE WHEN won_point THEN 1.0 ELSE 0.0 END) as overall_win_rate,
        AVG(CASE WHEN is_high_pressure AND won_point THEN 1.0
                 WHEN is_high_pressure THEN 0.0
                 ELSE NULL END) as hp_win_rate
    FROM player_points
    GROUP BY player, tour
    HAVING SUM(CASE WHEN is_high_pressure THEN 1 ELSE 0 END) >= {min_hp_points}
    ORDER BY hp_win_rate DESC
    """

    df = con.execute(query).fetchdf()
    df["clutch_delta"] = df["hp_win_rate"] - df["overall_win_rate"]
    con.close()

    # Add country codes
    df = enrich_with_country(df)

    return {
        "players": df.to_dict(orient="records"),
        "threshold": threshold,
        "best_under_pressure": df.nlargest(10, "hp_win_rate").to_dict(orient="records"),
        "worst_under_pressure": df.nsmallest(10, "hp_win_rate").to_dict(orient="records")
    }


@app.get("/api/players/scatter")
def get_player_scatter_data(
    years: str = Query(..., description="Comma-separated years"),
    tour: str = Query("ATP"),
    tournament: str = Query("All"),
    min_points: int = Query(400),
    pressure_percentile: int = Query(25)
):
    """
    Get combined player stats for scatter plot visualization:
    - Consistency vs Clutchness (tennis ball bubble chart)
    - Overall Win % vs High Pressure Win %
    """
    con = get_db()
    years_list = [int(y.strip()) for y in years.split(",")]
    years_str = ",".join(map(str, years_list))

    # Get importance threshold for high pressure
    threshold_query = f"""
    SELECT PERCENTILE_CONT({1 - pressure_percentile/100}) WITHIN GROUP (ORDER BY importance) as threshold
    FROM point_probability pp
    JOIN match_detail m USING(match_id)
    WHERE m.year IN ({years_str})
      AND ('{tour}' = 'All' OR m.tour = '{tour}')
    """
    threshold = con.execute(threshold_query).fetchone()[0]

    query = f"""
    WITH player_points AS (
        SELECT
            m.player1 as player,
            pp.p1_wp_delta as wp_delta,
            pp.importance,
            m.points_stake,
            pt.point_winner = 1 as won_point,
            pp.importance >= {threshold} as is_high_pressure,
            m.tour
        FROM point_detail pt
        JOIN match_detail m USING(match_id)
        JOIN point_probability pp USING(match_id, point_number)
        WHERE m.year IN ({years_str})
          AND ('{tour}' = 'All' OR m.tour = '{tour}')
          AND ('{tournament}' = 'All' OR m.tournament_name = '{tournament}')

        UNION ALL

        SELECT
            m.player2 as player,
            pp.p2_wp_delta as wp_delta,
            pp.importance,
            m.points_stake,
            pt.point_winner = 2 as won_point,
            pp.importance >= {threshold} as is_high_pressure,
            m.tour
        FROM point_detail pt
        JOIN match_detail m USING(match_id)
        JOIN point_probability pp USING(match_id, point_number)
        WHERE m.year IN ({years_str})
          AND ('{tour}' = 'All' OR m.tour = '{tour}')
          AND ('{tournament}' = 'All' OR m.tournament_name = '{tournament}')
    )
    SELECT
        player,
        tour,
        COUNT(*) as total_points,
        SUM(CASE WHEN is_high_pressure THEN 1 ELSE 0 END) as hp_points,

        -- Clutch score (EPA)
        SUM(wp_delta * importance * points_stake) as clutch_score,

        -- Consistency metrics
        AVG(wp_delta) as avg_delta,
        STDDEV_POP(wp_delta) as delta_std,
        CASE WHEN STDDEV_POP(wp_delta) > 0
             THEN AVG(wp_delta) / STDDEV_POP(wp_delta)
             ELSE 0 END as consistency_index,

        -- Win rates
        AVG(CASE WHEN won_point THEN 1.0 ELSE 0.0 END) as overall_win_rate,
        AVG(CASE WHEN is_high_pressure AND won_point THEN 1.0
                 WHEN is_high_pressure THEN 0.0
                 ELSE NULL END) as hp_win_rate
    FROM player_points
    GROUP BY player, tour
    HAVING COUNT(*) >= {min_points}
      AND SUM(CASE WHEN is_high_pressure THEN 1 ELSE 0 END) >= {min_points // 8}
    """

    df = con.execute(query).fetchdf()
    con.close()

    # Calculate derived metrics
    df["clutch_delta"] = df["hp_win_rate"] - df["overall_win_rate"]

    # Normalize consistency and clutch to percentiles for the scatter
    df["consistency_percentile"] = df["consistency_index"].rank(pct=True)
    df["clutch_percentile"] = df["clutch_score"].rank(pct=True)

    # Add country codes
    df = enrich_with_country(df)

    return {
        "players": df.to_dict(orient="records"),
        "threshold": threshold
    }


@app.get("/api/matches/unlikely")
def get_unlikely_matches(
    years: str = Query(...),
    tour: str = Query("ATP"),
    unlikely_threshold: float = Query(0.10, description="Win probability threshold for unlikely wins")
):
    """
    Get matches with unlikely comebacks.
    """
    con = get_db()
    years_list = [int(y.strip()) for y in years.split(",")]
    years_str = ",".join(map(str, years_list))

    query = f"""
    WITH match_probs AS (
        SELECT
            m.match_id,
            m.player1,
            m.player2,
            m.match_winner,
            m.tournament_name,
            m.year,
            MIN(pp.p1_win_prob_before) as min_p1_prob,
            MIN(1 - pp.p1_win_prob_before) as min_p2_prob
        FROM point_detail pt
        JOIN match_detail m USING(match_id)
        JOIN point_probability pp USING(match_id, point_number)
        WHERE m.year IN ({years_str})
          AND ('{tour}' = 'All' OR m.tour = '{tour}')
          AND m.match_winner IS NOT NULL
        GROUP BY m.match_id, m.player1, m.player2, m.match_winner, m.tournament_name, m.year
    )
    SELECT
        match_id,
        player1,
        player2,
        match_winner,
        tournament_name,
        year,
        CASE WHEN match_winner = player1 THEN min_p1_prob ELSE min_p2_prob END as lowest_win_prob
    FROM match_probs
    WHERE (match_winner = player1 AND min_p1_prob <= {unlikely_threshold})
       OR (match_winner = player2 AND min_p2_prob <= {unlikely_threshold})
    ORDER BY lowest_win_prob ASC
    LIMIT 20
    """

    df = con.execute(query).fetchdf()
    con.close()

    return {"matches": df.to_dict(orient="records")}


@app.get("/api/points/top")
def get_top_points(
    years: str = Query(...),
    tour: str = Query("ATP"),
    limit: int = Query(10)
):
    """
    Get the most important/pivotal points.
    """
    con = get_db()
    years_list = [int(y.strip()) for y in years.split(",")]
    years_str = ",".join(map(str, years_list))

    query = f"""
    SELECT
        m.match_id,
        m.player1,
        m.player2,
        m.tournament_name,
        m.year,
        pt.point_number,
        pt.score,
        pt.point_winner,
        pp.importance,
        pp.p1_win_prob_before,
        pp.p1_win_prob_if_p1_wins,
        pp.p1_win_prob_if_p2_wins,
        ABS(pp.p1_win_prob_if_p1_wins - pp.p1_win_prob_if_p2_wins) as swing
    FROM point_detail pt
    JOIN match_detail m USING(match_id)
    JOIN point_probability pp USING(match_id, point_number)
    WHERE m.year IN ({years_str})
      AND ('{tour}' = 'All' OR m.tour = '{tour}')
    ORDER BY swing DESC
    LIMIT {limit}
    """

    df = con.execute(query).fetchdf()
    con.close()

    return {"points": df.to_dict(orient="records")}


@app.get("/api/players/{player_name}")
def get_player_detail(
    player_name: str,
    years: str = Query(...)
):
    """
    Get detailed stats for a specific player.
    """
    con = get_db()
    years_list = [int(y.strip()) for y in years.split(",")]
    years_str = ",".join(map(str, years_list))

    query = f"""
    WITH player_matches AS (
        SELECT
            m.match_id,
            m.tournament_name,
            m.year,
            m.match_winner,
            CASE WHEN m.player1 = '{player_name}' THEN 1 ELSE 2 END as player_num
        FROM match_detail m
        WHERE (m.player1 = '{player_name}' OR m.player2 = '{player_name}')
          AND m.year IN ({years_str})
    )
    SELECT
        COUNT(DISTINCT match_id) as matches_played,
        SUM(CASE WHEN match_winner = '{player_name}' THEN 1 ELSE 0 END) as matches_won,
        COUNT(DISTINCT tournament_name) as tournaments
    FROM player_matches
    """

    stats = con.execute(query).fetchdf().to_dict(orient="records")[0]
    con.close()

    return {
        "player": player_name,
        "stats": stats
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
