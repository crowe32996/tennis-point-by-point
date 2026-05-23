"""
Centralized configuration for tennis simulation parameters and thresholds.
"""

# Minimum points per year thresholds for player filtering
MIN_POINTS_PER_YEAR = {
    "ATP": 400,
    "WTA": 200,
}

# Minimum high-pressure points per year for clutch analysis
MIN_HIGH_PRESSURE_POINTS_PER_YEAR = {
    "ATP": 50,
    "WTA": 30,
}

# High pressure is defined as the top percentile of point importance
HIGH_PRESSURE_PERCENTILE = 25

# Simulation parameters
N_SIMULATIONS_PER_SCENARIO = 1000
N_SIMULATIONS_PER_POINT = 3000  # 3 scenarios x 1000 each

# Unlikely win/loss probability thresholds
UNLIKELY_WIN_THRESHOLD = 0.10   # Win prob <= 10% and still won
UNLIKELY_LOSS_THRESHOLD = 0.90  # Win prob >= 90% and still lost

# Round points mapping (ATP/WTA Grand Slam points at stake)
ROUND_POINTS_MAP = {
    1: 10,    # R128
    2: 45,    # R64
    3: 90,    # R32
    4: 180,   # R16
    5: 360,   # QF
    6: 720,   # SF
    7: 1200,  # F
    8: 2000,  # Winner
}

# Tournament name mappings
TOURNAMENTS_MAP = {
    "ausopen": "Australian Open",
    "frenchopen": "French Open",
    "wimbledon": "Wimbledon",
    "usopen": "US Open",
}
