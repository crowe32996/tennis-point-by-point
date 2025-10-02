import duckdb
from pathlib import Path

# Base directory is the repo root
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed
if not (BASE_DIR / "data").exists():
    BASE_DIR = Path(__file__).resolve().parent  # fallback for cloud

DUCKDB_FILE = BASE_DIR / "outputs" / "sim_results.duckdb"
CSV_FILE = BASE_DIR / "outputs" / "all_points_with_importance.csv"
TABLE_NAME = "importance_results"

def transform_tables():
    con = duckdb.connect(DUCKDB_FILE)

    # Match-level table
    con.execute("""
    CREATE OR REPLACE TABLE match_detail AS
    SELECT 
        ROW_NUMBER() OVER (ORDER BY match_id) AS match_key,
        match_id,
        CASE
            WHEN match_id LIKE '%-MS%' THEN 'ATP'
            WHEN match_id LIKE '%-WS%' THEN 'WTA'
            WHEN LENGTH(match_id) >= 4 AND substr(split_part(match_id, '-', 3), 1, 1) = '1' THEN 'ATP'
            WHEN LENGTH(match_id) >= 4 AND substr(split_part(match_id, '-', 3), 1, 1) = '2' THEN 'WTA'
            ELSE 'Unknown'
        END AS tour,
        CASE tournament
            WHEN 'usopen' THEN 'US Open'
            WHEN 'wimbledon' THEN 'Wimbledon'
            WHEN 'ausopen' THEN 'Australian Open'
            WHEN 'frenchopen' THEN 'French Open'
            ELSE tournament
        END AS tournament_name,
        year, 
        player1, 
        player2, 
        best_of_5, 
        round, 
        points_stake, 
        match_winner
    FROM (
        SELECT DISTINCT match_id, tournament, year, player1, player2, best_of_5, round, points_stake, match_winner
        FROM importance_results
    ) AS distinct_matches;
    """)

    # Point-level table
    con.execute("""
    CREATE OR REPLACE TABLE point_detail AS
    SELECT 
        ROW_NUMBER() OVER (ORDER BY match_id, PointNumber) AS point_id,
        match_id, 
        PointNumber AS point_number, 
        PointWinner AS point_winner, 
        PointServer AS point_server,
        P1_Sets_Won AS p1_sets_won, 
        P2_Sets_Won AS p2_sets_won, 
        P1_Games_Won AS p1_games_won, 
        P2_Games_Won AS p2_games_won, 
        P1Score_Pre AS p1_score, 
        P2Score_Pre AS p2_score, 
        CASE 
            WHEN PointServer = 1 THEN CAST(P1Score_Pre AS VARCHAR) || '-' || CAST(P2Score_Pre AS VARCHAR)
            WHEN PointServer = 2 THEN CAST(P2Score_Pre AS VARCHAR) || '-' || CAST(P1Score_Pre AS VARCHAR)
            ELSE CAST(P1Score_Pre AS VARCHAR) || '-' || CAST(P2Score_Pre AS VARCHAR)
        END AS score,
        server_name, 
        returner_name, 
        is_tiebreak 
    FROM importance_results
    """)

    # Point probability table
    con.execute("""
    CREATE OR REPLACE TABLE point_probability AS
    SELECT 
        ROW_NUMBER() OVER (ORDER BY match_id, PointNumber) AS point_id,
        match_id, 
        PointNumber AS point_number, 
        p1_win_prob_before, 
        p1_win_prob_if_p1_wins, 
        p1_win_prob_if_p2_wins, 
        importance, 
        p1_wp_delta, 
        p2_wp_delta 
    FROM importance_results
    """)

    # Player dimension table
    con.execute("""
    CREATE OR REPLACE TABLE player_detail AS
        WITH all_players AS (
            SELECT player1 AS player, 
                CASE
                    WHEN match_id LIKE '%-MS%' THEN 'ATP'
                    WHEN match_id LIKE '%-WS%' THEN 'WTA'
                    WHEN LENGTH(match_id) >= 4 AND substr(split_part(match_id, '-', 3), 1, 1) = '1' THEN 'ATP'
                    WHEN LENGTH(match_id) >= 4 AND substr(split_part(match_id, '-', 3), 1, 1) = '2' THEN 'WTA'
                    ELSE 'Unknown'
                END AS tour,
                player1_serve_point_win_pct AS serve_pct,
                player1_return_point_win_pct AS return_pct,
                year
            FROM importance_results
            UNION
            SELECT player2 AS player, 
                CASE
                    WHEN match_id LIKE '%-MS%' THEN 'ATP'
                    WHEN match_id LIKE '%-WS%' THEN 'WTA'
                    WHEN LENGTH(match_id) >= 4 AND substr(split_part(match_id, '-', 3), 1, 1) = '1' THEN 'ATP'
                    WHEN LENGTH(match_id) >= 4 AND substr(split_part(match_id, '-', 3), 1, 1) = '2' THEN 'WTA'
                    ELSE 'Unknown'
                END AS tour,
                player2_serve_point_win_pct AS serve_pct,
                player2_return_point_win_pct AS return_pct,
                year
            FROM importance_results
        ),
        distinct_players AS (
            SELECT DISTINCT player
            FROM all_players
        ),
        most_recent_year AS (
            SELECT MAX(year) AS year FROM importance_results
        )
        SELECT 
            ROW_NUMBER() OVER () AS player_id,
            p.player,
            a.tour,
            a.serve_pct,
            a.return_pct,
            CASE WHEN MAX(a.year) = (SELECT year FROM most_recent_year) THEN 'Active' ELSE 'Inactive' END AS player_status
        FROM all_players a
        JOIN distinct_players p USING (player)
        GROUP BY p.player, a.tour, a.serve_pct, a.return_pct;

    """)

    con.close()
    return "Transform Completed"
