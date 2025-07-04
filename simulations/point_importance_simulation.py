import random
import pandas as pd
import re

### Simulation Core ###

from numba import njit, prange
import numpy as np

@njit
def simulate_game(server_win_prob, returner_win_prob, server, p1_points, p2_points):
    while True:
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
def simulate_tiebreak(server1_wp, returner2_wp, server2_wp, returner1_wp, starting_server):
    p1_tb_points = 0
    p2_tb_points = 0
    total_points_played = 0
    server = starting_server

    while True:
        if server == 1:
            win_prob = 0.5 * (server1_wp + (1.0 - returner2_wp))
            if np.random.random() < win_prob:
                p1_tb_points += 1
            else:
                p2_tb_points += 1
        else:
            win_prob = 0.5 * (server2_wp + (1.0 - returner1_wp))
            if np.random.random() < win_prob:
                p2_tb_points += 1
            else:
                p1_tb_points += 1

        total_points_played += 1

        # Check for winner
        if (p1_tb_points >= 7 or p2_tb_points >= 7) and abs(p1_tb_points - p2_tb_points) >= 2:
            return 1 if p1_tb_points > p2_tb_points else 2

        # Update server: switch after first point, then every two
        if total_points_played == 1 or (total_points_played > 1 and (total_points_played - 1) % 4 == 0):
            server = 3 - server

@njit
def simulate_set(server1_wp, returner2_wp, server2_wp, returner1_wp,
                 p1_games, p2_games, p1_points, p2_points,
                 starting_server, in_progress_game):
    server = starting_server

    if in_progress_game:
        winner = simulate_game(
            server1_wp if server == 1 else server2_wp,
            returner2_wp if server == 1 else returner1_wp,
            server, p1_points, p2_points
        )
        if winner == 1:
            p1_games += 1
        else:
            p2_games += 1
        server = 3 - server

    while True:
        # If 6-6, simulate tiebreak
        if p1_games == 6 and p2_games == 6:
            winner = simulate_tiebreak(server1_wp, returner2_wp, server2_wp, returner1_wp, server)
            if winner == 1:
                p1_games += 1
            else:
                p2_games += 1
            return winner

        # Regular game
        winner = simulate_game(
            server1_wp if server == 1 else server2_wp,
            returner2_wp if server == 1 else returner1_wp,
            server, 0, 0
        )
        if winner == 1:
            p1_games += 1
        else:
            p2_games += 1

        # Win by 2 rule
        if p1_games >= 6 and p1_games - p2_games >= 2:
            return 1
        if p2_games >= 6 and p2_games - p1_games >= 2:
            return 2

        server = 3 - server

@njit
def simulate_match(server1_wp, returner2_wp, server2_wp, returner1_wp,
                   p1_sets, p2_sets,
                   p1_games, p2_games,
                   p1_points, p2_points,
                   starting_server, in_progress_game,
                   best_of):
    server = starting_server

    while p1_sets < (best_of // 2 + 1) and p2_sets < (best_of // 2 + 1):
        winner = simulate_set(
            server1_wp, returner2_wp, server2_wp, returner1_wp,
            p1_games, p2_games, p1_points, p2_points,
            server, in_progress_game
        )
        if winner == 1:
            p1_sets += 1
        else:
            p2_sets += 1

        p1_games = p2_games = p1_points = p2_points = 0
        in_progress_game = False
        server = 3 - server

    return 1 if p1_sets > p2_sets else 2
### Importance Calculation ###

def monte_carlo_win_prob_from_state(server1_wp, returner2_wp, server2_wp, returner1_wp,
                                    p1_sets, p2_sets,
                                    p1_games, p2_games,
                                    p1_points, p2_points,
                                    starting_server, in_progress_game,
                                    n_simulations=50):
    p1_wins = 0
    for _ in range(n_simulations):
        result = simulate_match(
            server1_wp, returner2_wp, server2_wp, returner1_wp,
            p1_sets, p2_sets,
            p1_games, p2_games,
            p1_points, p2_points,
            starting_server, in_progress_game,
            5  # best_of 5 match
        )
        if result == 1:
            p1_wins += 1
    win_prob = p1_wins / n_simulations
    return win_prob, 1 - win_prob

def compute_point_importance(server1_wp, returner2_wp, server2_wp, returner1_wp,
                             p1_sets, p2_sets,
                             p1_games, p2_games,
                             p1_points, p2_points,
                             starting_server, in_progress_game,
                             n_simulations=50):
    p1_before, _ = monte_carlo_win_prob_from_state(
        server1_wp, returner2_wp, server2_wp, returner1_wp,
        p1_sets, p2_sets, p1_games, p2_games,
        p1_points, p2_points, starting_server, in_progress_game,
        n_simulations
    )
    p1_win = monte_carlo_win_prob_from_state(
        server1_wp, returner2_wp, server2_wp, returner1_wp,
        p1_sets, p2_sets, p1_games, p2_games,
        p1_points + 1, p2_points, starting_server, True,
        n_simulations
    )[0]
    p1_lose = monte_carlo_win_prob_from_state(
        server1_wp, returner2_wp, server2_wp, returner1_wp,
        p1_sets, p2_sets, p1_games, p2_games,
        p1_points, p2_points + 1, starting_server, True,
        n_simulations
    )[0]
    return {
        'p1_win_prob_before': p1_before,
        'p1_win_prob_if_p1_wins': p1_win,
        'p1_win_prob_if_p2_wins': p1_lose,
        'importance': abs(p1_win - p1_lose)
    }

def tennis_score_to_int(score, is_tiebreak=False):
    if is_tiebreak:
        try:
            return int(score)
        except:
            return 0
    return {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}.get(score, 0)

### Tracking Match State and Applying Importance ###


def importance_row_fn(row, n_simulations=100):
    try:
        score_str = str(row['score'])

        # Detect if we're in a tiebreak
        if (row['P1_Games_Won'] == 6 and row['P2_Games_Won'] == 6):
            is_tiebreak = True
            match = re.match(r"(\d+)-(\d+)", score_str)
            if not match:
                raise ValueError(f"Invalid tiebreak score format: {score_str}")
            server_points, returner_points = map(int, match.groups())
        else:
            is_tiebreak = False
            match = re.match(r"(0|15|30|40|AD)-(0|15|30|40|AD)", score_str)
            if not match:
                raise ValueError(f"Invalid standard score format: {score_str}")
            server_score_str, returner_score_str = match.groups()
            score_map = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
            server_points = score_map.get(server_score_str)
            returner_points = score_map.get(returner_score_str)

        if row['PointServer'] == 1:
            server1_wp = row['player1_serve_point_win_pct']
            returner2_wp = row['player2_return_point_win_pct']
            server2_wp = row['player2_serve_point_win_pct']
            returner1_wp = row['player1_return_point_win_pct']
            p1_sets = row['P1_Sets_Won']
            p2_sets = row['P2_Sets_Won']
            p1_games = row['P1_Games_Won']
            p2_games = row['P2_Games_Won']
        else:
            server1_wp = row['player2_serve_point_win_pct']
            returner2_wp = row['player1_return_point_win_pct']
            server2_wp = row['player1_serve_point_win_pct']
            returner1_wp = row['player2_return_point_win_pct']
            p1_sets = row['P2_Sets_Won']
            p2_sets = row['P1_Sets_Won']
            p1_games = row['P2_Games_Won']
            p2_games = row['P1_Games_Won']

        result = compute_point_importance(
            server1_wp, returner2_wp, server2_wp, returner1_wp,
            p1_sets, p2_sets,
            p1_games, p2_games,
            server_points, returner_points,
            row['PointServer'], in_progress_game=True,
            n_simulations=n_simulations
        )
        return pd.Series(result)

    except Exception as e:
        print(f"Row {row.name} error: {e}")
        return pd.Series({
            'p1_win_prob_before': None,
            'p1_win_prob_if_p1_wins': None,
            'p1_win_prob_if_p2_wins': None,
            'importance': None
        })

def compute_importance_for_df(df, n_simulations=100):
    results = df.apply(lambda row: importance_row_fn(row, n_simulations), axis=1)
    df = df.join(results)
    return df