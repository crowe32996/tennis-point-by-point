import re
import pandas as pd
import numpy as np
from numba import njit
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType

# --- Numba-accelerated simulation functions ---
@njit
def simulate_game(p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp,
                  server, p1_points, p2_points):
    while True:
        if server == 1:
            server_win_prob = p1_serve_wp
            returner_win_prob = p2_return_wp
        else:
            server_win_prob = p2_serve_wp
            returner_win_prob = p1_return_wp

        win_prob = 0.5 * (server_win_prob + (1.0 - returner_win_prob))
        if np.random.random() < win_prob:
            if server == 1:
                p1_points += 1
            else:
                p2_points += 1
        else:
            if server == 1:
                p2_points += 1
            else:
                p1_points += 1

        if p1_points >= 4 and p1_points - p2_points >= 2:
            return 1
        if p2_points >= 4 and p2_points - p1_points >= 2:
            return 2

@njit
def simulate_tiebreak(p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp, starting_server):
    p1_tb_points = 0
    p2_tb_points = 0
    total_points_played = 0
    server = starting_server
    while True:
        if server == 1:
            server_win_prob = p1_serve_wp
            returner_win_prob = p2_return_wp
        else:
            server_win_prob = p2_serve_wp
            returner_win_prob = p1_return_wp

        win_prob = 0.5 * (server_win_prob + (1.0 - returner_win_prob))
        if np.random.random() < win_prob:
            if server == 1:
                p1_tb_points += 1
            else:
                p2_tb_points += 1
        else:
            if server == 1:
                p2_tb_points += 1
            else:
                p1_tb_points += 1

        total_points_played += 1
        if (p1_tb_points >= 7 or p2_tb_points >= 7) and abs(p1_tb_points - p2_tb_points) >= 2:
            return 1 if p1_tb_points > p2_tb_points else 2

        # Change server: first point same, then every two points
        if total_points_played == 1 or (total_points_played > 1 and (total_points_played - 1) % 4 == 0):
            server = 3 - server

def monte_carlo_win_prob_from_state(p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp,
                                    p1_sets, p2_sets, p1_games, p2_games,
                                    p1_points, p2_points, starting_server,
                                    in_progress_game, best_of, n_simulations=50):
    p1_match_wins = 0
    sets_to_win = 3 if best_of == 5 else 2

    for _ in range(n_simulations):
        # Copy scores for simulation
        sim_p1_sets = p1_sets
        sim_p2_sets = p2_sets
        sim_p1_games = p1_games
        sim_p2_games = p2_games
        sim_p1_points = p1_points
        sim_p2_points = p2_points
        server = starting_server

        while sim_p1_sets < sets_to_win and sim_p2_sets < sets_to_win:

            # Tiebreak if both players have 6 games
            if sim_p1_games == 6 and sim_p2_games == 6:
                winner = simulate_tiebreak(p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp, server)
            else:
                winner = simulate_game(p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp,
                                       server, sim_p1_points, sim_p2_points)

            # Update game score
            if winner == 1:
                sim_p1_games += 1
            else:
                sim_p2_games += 1

            # Reset points for next game
            sim_p1_points = 0
            sim_p2_points = 0

            # Check for set win (non-tiebreak)
            if sim_p1_games >= 6 and sim_p1_games - sim_p2_games >= 2:
                sim_p1_sets += 1
                sim_p1_games = 0
                sim_p2_games = 0
            elif sim_p2_games >= 6 and sim_p2_games - sim_p1_games >= 2:
                sim_p2_sets += 1
                sim_p1_games = 0
                sim_p2_games = 0

            # Alternate server
            server = 2 if server == 1 else 1

        if sim_p1_sets == sets_to_win:
            p1_match_wins += 1

    p1_win_prob = p1_match_wins / n_simulations
    return p1_win_prob, 1 - p1_win_prob



def compute_point_importance(server1_wp, returner2_wp, server2_wp, returner1_wp,
                             p1_sets, p2_sets, p1_games, p2_games,
                             p1_points, p2_points, starting_server, in_progress_game,
                             best_of, n_simulations=50):
    p1_before, _ = monte_carlo_win_prob_from_state(
        server1_wp, returner2_wp, server2_wp, returner1_wp,
        p1_sets, p2_sets, p1_games, p2_games,
        p1_points, p2_points, starting_server, in_progress_game, best_of,
        n_simulations
    )
    p1_win, _ = monte_carlo_win_prob_from_state(
        server1_wp, returner2_wp, server2_wp, returner1_wp,
        p1_sets, p2_sets, p1_games, p2_games,
        p1_points + 1, p2_points, starting_server, True, best_of, n_simulations
    )
    p1_lose, _ = monte_carlo_win_prob_from_state(
        server1_wp, returner2_wp, server2_wp, returner1_wp,
        p1_sets, p2_sets, p1_games, p2_games,
        p1_points, p2_points + 1, starting_server, True, best_of, n_simulations
    )
    return {
        'p1_win_prob_before': p1_before,
        'p1_win_prob_if_p1_wins': p1_win,
        'p1_win_prob_if_p2_wins': p1_lose,
        'importance': abs(p1_win - p1_lose)
    }

# --- Row-wise function ---
def importance_row_fn(row, n_simulations=100):
    try:
        # Determine points in current game
        score_str = str(row['score'])
        is_tiebreak = (row['P1_Games_Won'] == 6 and row['P2_Games_Won'] == 6)
        if is_tiebreak:
            match = re.match(r"(\d+)-(\d+)", score_str)
            p1_points, p2_points = map(int, match.groups())
        else:
            score_map = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
            match = re.match(r"(0|15|30|40|AD)-(0|15|30|40|AD)", score_str)
            p1_score_str, p2_score_str = match.groups()
            p1_points = score_map.get(p1_score_str, 0)
            p2_points = score_map.get(p2_score_str, 0)

        p1_sets = row['P1_Sets_Won']
        p2_sets = row['P2_Sets_Won']
        p1_games = row['P1_Games_Won']
        p2_games = row['P2_Games_Won']

        p1_serve_wp = row['player1_serve_point_win_pct']
        p1_return_wp = row['player1_return_point_win_pct']
        p2_serve_wp = row['player2_serve_point_win_pct']
        p2_return_wp = row['player2_return_point_win_pct']

        starting_server = row['PointServer']  # 1 or 2
        
        best_of = row['best_of_5']

        result = compute_point_importance(
            p1_serve_wp, p1_return_wp,
            p2_serve_wp, p2_return_wp,
            p1_sets, p2_sets,
            p1_games, p2_games,
            p1_points, p2_points,
            starting_server, in_progress_game=True,
            best_of = best_of,
            n_simulations=n_simulations
        )
        result['importance'] *= row.get('round_weight', 1)
        return pd.Series([
            result['p1_win_prob_before'],
            result['p1_win_prob_if_p1_wins'],
            result['p1_win_prob_if_p2_wins'],
            result['importance']
        ])

    except Exception as e:
        return pd.Series({
            'p1_win_prob_before': None,
            'p1_win_prob_if_p1_wins': None,
            'p1_win_prob_if_p2_wins': None,
            'importance': None
        })

# --- Schema for Spark UDF ---
importance_schema = StructType([
    StructField("p1_win_prob_before", DoubleType(), True),
    StructField("p1_win_prob_if_p1_wins", DoubleType(), True),
    StructField("p1_win_prob_if_p2_wins", DoubleType(), True),
    StructField("importance", DoubleType(), True),
])

# --- Scalar Pandas UDF for row-wise Spark application ---
#@pandas_udf(importance_schema)
#def importance_udf(pdf: pd.DataFrame) -> pd.DataFrame:
#    # pdf is a Pandas DataFrame with all columns of your Spark DataFrame
#    return pdf.apply(lambda row: importance_row_fn(row, n_simulations=100), axis=1)

def importance_batch_fn(pdf: pd.DataFrame, n_simulations: int) -> pd.DataFrame:
    results = pdf.apply(
        lambda row: importance_row_fn(row, n_simulations=n_simulations),
        axis=1
    )
    # results is already a DataFrame with proper columns (from importance_row_fn)
    results.columns = ["p1_win_prob_before", "p1_win_prob_if_p1_wins",
                       "p1_win_prob_if_p2_wins", "importance"]
    return results
