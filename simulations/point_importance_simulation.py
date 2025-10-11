import numpy as np
import pandas as pd
from numba import njit, prange
from pyspark.sql.types import StructType, StructField, DoubleType

# -----------------------------
# Numba batch simulation functions
# -----------------------------

@njit
def simulate_game_batch(n_simulations, p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp,
                        p1_points, p2_points, server):
    """
    Simulate multiple games in parallel using arrays for point scores.
    Returns an array of winners (1 or 2) of shape (n_simulations,)
    """
    winners = np.zeros(n_simulations, dtype=np.int8)
    p1_pts = np.full(n_simulations, p1_points)
    p2_pts = np.full(n_simulations, p2_points)
    serv = np.full(n_simulations, server)

    for i in range(n_simulations):
        while True:
            if serv[i] == 1:
                server_win_prob = p1_serve_wp
                returner_win_prob = p2_return_wp
            else:
                server_win_prob = p2_serve_wp
                returner_win_prob = p1_return_wp

            win_prob = 0.5 * (server_win_prob + (1.0 - returner_win_prob))
            if np.random.random() < win_prob:
                if serv[i] == 1:
                    p1_pts[i] += 1
                else:
                    p2_pts[i] += 1
            else:
                if serv[i] == 1:
                    p2_pts[i] += 1
                else:
                    p1_pts[i] += 1

            if p1_pts[i] >= 4 and p1_pts[i] - p2_pts[i] >= 2:
                winners[i] = 1
                break
            if p2_pts[i] >= 4 and p2_pts[i] - p1_pts[i] >= 2:
                winners[i] = 2
                break

    return winners

@njit
def simulate_tiebreak_batch(n_simulations, p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp, starting_server):
    winners = np.zeros(n_simulations, dtype=np.int8)
    p1_pts = np.zeros(n_simulations, dtype=np.int8)
    p2_pts = np.zeros(n_simulations, dtype=np.int8)
    serv = np.full(n_simulations, starting_server)
    total_points = np.zeros(n_simulations, dtype=np.int8)

    for i in range(n_simulations):
        while True:
            if serv[i] == 1:
                server_win_prob = p1_serve_wp
                returner_win_prob = p2_return_wp
            else:
                server_win_prob = p2_serve_wp
                returner_win_prob = p1_return_wp

            win_prob = 0.5 * (server_win_prob + (1.0 - returner_win_prob))
            if np.random.random() < win_prob:
                if serv[i] == 1:
                    p1_pts[i] += 1
                else:
                    p2_pts[i] += 1
            else:
                if serv[i] == 1:
                    p2_pts[i] += 1
                else:
                    p1_pts[i] += 1

            total_points[i] += 1

            # Check tiebreak end
            if (p1_pts[i] >= 7 or p2_pts[i] >= 7) and abs(p1_pts[i] - p2_pts[i]) >= 2:
                winners[i] = 1 if p1_pts[i] > p2_pts[i] else 2
                break

            # Change server: first point same, then every 2 points
            if total_points[i] == 1 or (total_points[i] > 1 and (total_points[i]-1) % 4 == 0):
                serv[i] = 3 - serv[i]

    return winners

# -----------------------------
# Monte Carlo match simulation (vectorized)
# -----------------------------
@njit
def monte_carlo_win_prob_batch(n_simulations, p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp,
                               p1_sets, p2_sets, p1_games, p2_games,
                               p1_points, p2_points, starting_server,
                               best_of):
    sets_to_win = 3 if best_of == 5 else 2
    p1_match_wins = 0

    for sim in range(n_simulations):
        sim_p1_sets = p1_sets
        sim_p2_sets = p2_sets
        sim_p1_games = p1_games
        sim_p2_games = p2_games
        sim_p1_points = p1_points
        sim_p2_points = p2_points
        server = starting_server

        while sim_p1_sets < sets_to_win and sim_p2_sets < sets_to_win:
            # Tiebreak if needed
            if sim_p1_games == 6 and sim_p2_games == 6:
                winner = simulate_tiebreak_batch(1, p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp, server)[0]
            else:
                winner = simulate_game_batch(1, p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp,
                                             sim_p1_points, sim_p2_points, server)[0]

            if winner == 1:
                sim_p1_games += 1
            else:
                sim_p2_games += 1

            # Reset points
            sim_p1_points = 0
            sim_p2_points = 0

            # Check for set win
            if sim_p1_games >= 6 and sim_p1_games - sim_p2_games >= 2:
                sim_p1_sets += 1
                sim_p1_games = 0
                sim_p2_games = 0
            elif sim_p2_games >= 6 and sim_p2_games - sim_p1_games >= 2:
                sim_p2_sets += 1
                sim_p1_games = 0
                sim_p2_games = 0

            # Alternate server
            server = 3 - server

        if sim_p1_sets == sets_to_win:
            p1_match_wins += 1

    return p1_match_wins / n_simulations

# -----------------------------
# Point importance computation
# -----------------------------
def compute_point_importance_batch(p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp,
                                   p1_sets, p2_sets, p1_games, p2_games,
                                   p1_points, p2_points, starting_server,
                                   best_of, n_simulations=50):
    # Current state
    p1_before = monte_carlo_win_prob_batch(n_simulations, p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp,
                                           p1_sets, p2_sets, p1_games, p2_games,
                                           p1_points, p2_points, starting_server,
                                           best_of)
    # If P1 wins next point
    p1_win = monte_carlo_win_prob_batch(n_simulations, p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp,
                                        p1_sets, p2_sets, p1_games, p2_games,
                                        p1_points + 1, p2_points, starting_server,
                                        best_of)
    # If P2 wins next point
    p1_lose = monte_carlo_win_prob_batch(n_simulations, p1_serve_wp, p1_return_wp, p2_serve_wp, p2_return_wp,
                                         p1_sets, p2_sets, p1_games, p2_games,
                                         p1_points, p2_points + 1, starting_server,
                                         best_of)
    importance = abs(p1_win - p1_lose)
    return p1_before, p1_win, p1_lose, importance

# -----------------------------
# Batch UDF for Spark
# -----------------------------
importance_schema = StructType([
    StructField("p1_win_prob_before", DoubleType(), True),
    StructField("p1_win_prob_if_p1_wins", DoubleType(), True),
    StructField("p1_win_prob_if_p2_wins", DoubleType(), True),
    StructField("importance", DoubleType(), True),
])

def importance_batch_fn(pdf: pd.DataFrame, n_simulations=50) -> pd.DataFrame:
    results = np.zeros((len(pdf), 4))
    for i, row in enumerate(pdf.itertuples(index=False)):
        p1_before, p1_win, p1_lose, importance = compute_point_importance_batch(
            row.player1_serve_point_win_pct,
            row.player1_return_point_win_pct,
            row.player2_serve_point_win_pct,
            row.player2_return_point_win_pct,
            row.P1_Sets_Won,
            row.P2_Sets_Won,
            row.P1_Games_Won,
            row.P2_Games_Won,
            row.p1_points,
            row.p2_points,
            row.PointServer,
            row.best_of_5,
            n_simulations
        )
        results[i, :] = [p1_before, p1_win, p1_lose, importance]
    return pd.DataFrame(results, columns=["p1_win_prob_before", "p1_win_prob_if_p1_wins",
                                          "p1_win_prob_if_p2_wins", "importance"])