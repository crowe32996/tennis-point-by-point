# Databricks notebook source

import numpy
!pip install numba
import numba



# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, col

points_df = spark.read.option("header", "true").csv("/Volumes/tennis_catalog/raw_data/raw_tennis_vol/*-points.csv")
matches_df = spark.read.option("header", "true").csv("/Volumes/tennis_catalog/raw_data/raw_tennis_vol/*-matches.csv")

points_df = points_df.filter((col("PointWinner") != 0) & (col("PointServer") != 0))
points_df = points_df.withColumn("point_id", monotonically_increasing_id())

print(f"Matches rows: {matches_df.count()}")
print(f"Points rows: {points_df.count()}")


# COMMAND ----------

from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from pyspark.sql.functions import monotonically_increasing_id

def standardize_name(full_name):
    if not full_name:
        return None
    parts = full_name.strip().split()
    if len(parts) < 2:
        return None
    first_name = parts[0]
    last_name = " ".join(parts[1:])
    initial = first_name[0]
    return f"{initial}. {last_name}"

standardize_name_udf = udf(standardize_name, StringType())

# Combine all unique player names from player1 and player2
all_players_df = matches_df.select("player1").union(matches_df.select("player2")).distinct()\
    .withColumnRenamed("player1", "player_name")

# Standardize names
dim_player_df = all_players_df.withColumn("player_name_clean", standardize_name_udf(col("player_name")))\
    .dropna(subset=["player_name_clean"])\
    .dropDuplicates(["player_name_clean"])\
    .withColumn("player_id", monotonically_increasing_id())\
    .select("player_id", "player_name", "player_name_clean")

dim_player_df.write.format("delta").mode("overwrite").saveAsTable("dim_player")


# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

dim_event_df = matches_df.select("slam", "year").distinct()
dim_event_df = dim_event_df.withColumn("event_id", monotonically_increasing_id())
dim_event_df = dim_event_df.select("event_id", "slam", "year")
dim_event_df.write.format("delta").mode("overwrite").saveAsTable("dim_event")


# COMMAND ----------

from pyspark.sql.functions import col, monotonically_increasing_id, udf, expr
from pyspark.sql.types import StringType

# Your existing standardize_name function and UDF
def standardize_name(full_name):
    if not full_name:
        return None
    parts = full_name.strip().split()
    if len(parts) < 2:
        return None
    first_name = parts[0]
    last_name = " ".join(parts[1:])
    initial = first_name[0]
    return f"{initial}. {last_name}"

standardize_name_udf = udf(standardize_name, StringType())

# Load the reference tables
dim_event_df = spark.table("dim_event")
dim_player_df = spark.table("dim_player")

# Step 1: Add cleaned player name columns to matches_df
matches_df_cleaned = matches_df \
    .withColumn("player1_clean", standardize_name_udf(col("player1"))) \
    .withColumn("player2_clean", standardize_name_udf(col("player2")))

# Step 2: Join matches to events
matches_with_event = matches_df_cleaned.join(
    dim_event_df,
    on=["slam", "year"],
    how="left"
)

# Step 3: Join to get player1_id and player2_id from dim_player using cleaned names
matches_with_players = matches_with_event \
    .join(
        dim_player_df.select(col("player_id").alias("player1_id"), col("player_name_clean").alias("player1_clean")),
        on="player1_clean",
        how="left"
    ) \
    .join(
        dim_player_df.select(col("player_id").alias("player2_id"), col("player_name_clean").alias("player2_clean")),
        on="player2_clean",
        how="left"
    )

# Step 4: Select columns and add tour based on match_id suffix (4-digit number at end)
dim_match_df = matches_with_players.select(
    "match_id", "event_id", "player1_id", "player2_id", "round"
).withColumn(
    "tour",
    expr("""
        CASE 
          WHEN substring(match_id, -4, 1) = '1' THEN 'ATP'
          WHEN substring(match_id, -4, 1) = '2' THEN 'WTA'
          WHEN substring(match_id, -5, 2) = 'MS' THEN 'ATP'
          WHEN substring(match_id, -5, 2) = 'WS' THEN 'WTA'
          ELSE 'UNKNOWN'
        END
    """)
)


# Step 5: Add surrogate match_key
dim_match_df = dim_match_df.withColumn("match_key", monotonically_increasing_id())

# Step 6: Write to Delta table
dim_match_df.write.format("delta").mode("overwrite").saveAsTable("dim_match")


# COMMAND ----------

from pyspark.sql.functions import when, col, monotonically_increasing_id

# Load reference tables
dim_match_df = spark.table("dim_match")  # contains: match_id, match_key, player1_id, player2_id

# 1. Ensure you're joining on the correct column (match_id)
# If your points_df doesn't have match_id but has match_key, use that
points_joined = points_df.join(
    dim_match_df.select("match_id", "match_key", "player1_id", "player2_id"),
    on="match_id",
    how="inner"
)

# 2. Assign server_id, returner_id using PointServer (1 = player1, 2 = player2)
points_with_players = points_joined.withColumn(
    "server_id",
    when(col("PointServer") == 1, col("player1_id")).when(col("PointServer") == 2, col("player2_id"))
).withColumn(
    "returner_id",
    when(col("PointServer") == 1, col("player2_id")).when(col("PointServer") == 2, col("player1_id"))
)

# 3. Assign point_winner_id using PointWinner (1 = player1, 2 = player2)
points_with_players = points_with_players.withColumn(
    "point_winner_id",
    when(col("PointWinner") == 1, col("player1_id")).when(col("PointWinner") == 2, col("player2_id"))
)

# 4. Add surrogate key
fact_points_df = points_with_players.withColumn("point_id", monotonically_increasing_id())

# 5. Select desired fields
fact_points_df = fact_points_df.select(
    "point_id", "match_key", "PointNumber", "SetNo", "GameNo","PointWinner", "PointServer", "server_id", "returner_id",  "point_winner_id", "P1GamesWon", "P2GamesWon", "SetWinner", "P1Score", "P2Score", "P1PointsWon", "P2PointsWon"
    # Add any extra point-level columns you want to include
)

# 6. Write to Delta table
fact_points_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("fact_points")


# COMMAND ----------

from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Define a window over each match, ordered by point number
match_point_window = Window.partitionBy("match_key").orderBy("PointNumber")

# Shift game counts to represent the *prior* state before the current point
fact_points_df = fact_points_df.withColumn("P1GamesWon_Pre", F.lag("P1GamesWon").over(match_point_window))
fact_points_df = fact_points_df.withColumn("P2GamesWon_Pre", F.lag("P2GamesWon").over(match_point_window))

# For the first point in a match, fill nulls with 0
fact_points_df = fact_points_df.fillna({"P1GamesWon_Pre": 0, "P2GamesWon_Pre": 0})

fact_points_df = fact_points_df.withColumn("SetNo_Pre", F.lag("SetNo").over(match_point_window))
fact_points_df = fact_points_df.fillna({"SetNo_Pre": 1})


# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE default.dim_player
# MAGIC ADD COLUMNS (
# MAGIC   serve_points_won LONG,
# MAGIC   serve_points_total LONG,
# MAGIC   serve_point_win_pct DOUBLE,
# MAGIC   return_points_won LONG,
# MAGIC   return_points_total LONG,
# MAGIC   return_point_win_pct DOUBLE
# MAGIC )
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.getOrCreate()

# Load your source dataframes
fact_points_df = spark.table("fact_points")  # or however you load it

# 1. Flag if server won the point
fact_points_df = fact_points_df.withColumn(
    "server_point_win",
    F.expr("CASE WHEN server_id = point_winner_id THEN 1 ELSE 0 END")
)

# 2. Aggregate serve stats per player
serve_stats_df = fact_points_df.groupBy("server_id").agg(
    F.sum("server_point_win").alias("serve_points_won"),
    F.count("*").alias("serve_points_total")
).withColumn(
    "serve_point_win_pct",
    F.col("serve_points_won") / F.col("serve_points_total")
)

# 3. Flag return points won (inverse of server_point_win)
fact_points_df = fact_points_df.withColumn(
    "return_point_win",
    1 - F.col("server_point_win")
)

# 4. Aggregate return stats per player
return_stats_df = fact_points_df.groupBy("returner_id").agg(
    F.sum("return_point_win").alias("return_points_won"),
    F.count("*").alias("return_points_total")
).withColumn(
    "return_point_win_pct",
    F.col("return_points_won") / F.col("return_points_total")
)

# 5. Join serve and return stats by player id
player_stats_df = serve_stats_df.join(
    return_stats_df,
    serve_stats_df.server_id == return_stats_df.returner_id,
    how="outer"
).select(
    F.coalesce(serve_stats_df.server_id, return_stats_df.returner_id).alias("player_id"),
    "serve_points_won",
    "serve_points_total",
    "serve_point_win_pct",
    "return_points_won",
    "return_points_total",
    "return_point_win_pct"
)

# 6. Drop existing serve/return stat columns from dim_player to avoid ambiguity
cols_to_drop = [
    "serve_points_won",
    "serve_points_total",
    "serve_point_win_pct",
    "return_points_won",
    "return_points_total",
    "return_point_win_pct",
    "tour"
]

for col in cols_to_drop:
    if col in dim_player_df.columns:
        dim_player_df = dim_player_df.drop(col)

# 7. Join dim_player with the newly computed player stats
dim_player_enriched_df = dim_player_df.join(
    player_stats_df,
    on="player_id",
    how="left"
)

# 8. Fill missing stats with default averages
dim_player_enriched_df = dim_player_enriched_df.fillna({
    "serve_point_win_pct": 0.62,
    "return_point_win_pct": 0.38
})

# Load dim_match table
dim_match_df = spark.table("dim_match")

# Extract player-tour mappings from dim_match
player_tour_df = dim_match_df.select(
    F.col("player1_id").alias("player_id"),
    F.col("tour")
).union(
    dim_match_df.select(
        F.col("player2_id").alias("player_id"),
        F.col("tour")
    )
).distinct()

# Aggregate tours per player
player_tour_agg_df = player_tour_df.groupBy("player_id").agg(
    F.collect_set("tour").alias("tours")
)

# UDF to classify tours
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

def classify_tour(tours):
    if tours is None or len(tours) == 0:
        return None
    tours_set = set(tours)
    if tours_set == {"ATP"}:
        return "ATP"
    elif tours_set == {"WTA"}:
        return "WTA"
    else:
        return "Both"

classify_tour_udf = udf(classify_tour, StringType())

player_tour_classified_df = player_tour_agg_df.withColumn(
    "tour",
    classify_tour_udf(F.col("tours"))
).select("player_id", "tour")

# Join the tour info to your dim_player_enriched_df
dim_player_enriched_df = dim_player_enriched_df.join(
    player_tour_classified_df,
    on="player_id",
    how="left"
)

# Optionally fill missing tour with "Unknown" or null
dim_player_enriched_df = dim_player_enriched_df.fillna({"tour": "Unknown"})

# Then continue with step 8 (fillna for stats) and so on

# 9. Select and order columns as desired
output_columns = [
    "player_id",
    "player_name",
    "player_name_clean",
    "tour",
    "serve_points_won",
    "serve_points_total",
    "serve_point_win_pct",
    "return_points_won",
    "return_points_total",
    "return_point_win_pct"
]

dim_player_enriched_df = dim_player_enriched_df.select(*output_columns)

# 10. Overwrite dim_player table with enriched stats
dim_player_enriched_df.write.format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable("dim_player") 



# COMMAND ----------

from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Define a corrected window: order by numeric PointNumber
match_point_window = Window.partitionBy("match_key").orderBy(F.col("PointNumber").cast("int"))

# Add P1Score_Pre and P2Score_Pre using lag
fact_points_df = (
    fact_points_df
    .withColumn("P1Score_Pre", F.lag("P1Score").over(match_point_window))
    .withColumn("P2Score_Pre", F.lag("P2Score").over(match_point_window))
)

# Fill nulls in first row with 0s (start of game)
fact_points_df = (
    fact_points_df
    .fillna({"P1Score_Pre": 0, "P2Score_Pre": 0})
)

# Add is_tiebreak based on prior game score (i.e., 6-6)
fact_points_df = fact_points_df.withColumn(
    "is_tiebreak",
    (F.col("P1GamesWon_Pre") == 6) & (F.col("P2GamesWon_Pre") == 6)
)

# Shift game counts to represent the *prior* state before the current point
fact_points_df = fact_points_df.withColumn("P1GamesWon_Pre", F.lag("P1GamesWon").over(match_point_window))
fact_points_df = fact_points_df.withColumn("P2GamesWon_Pre", F.lag("P2GamesWon").over(match_point_window))

# For the first point in a match, fill nulls with 0
fact_points_df = fact_points_df.fillna({"P1GamesWon_Pre": 0, "P2GamesWon_Pre": 0})

# Lag prior SetNo and prior PointWinner
fact_points_df = fact_points_df \
    .withColumn("Prev_SetNo", F.lag("SetNo").over(match_point_window)) \
    .withColumn("Prev_PointWinner", F.lag("PointWinner").over(match_point_window))

# Identify where a new set starts (i.e., SetNo increases)
fact_points_df = fact_points_df.withColumn(
    "New_Set_Start",
    F.when((F.col("Prev_SetNo").isNotNull()) & (F.col("SetNo") > F.col("Prev_SetNo")), F.lit(1)).otherwise(F.lit(0))
)

# Set wins gained at the start of each new set — only 1 player can win the previous set
fact_points_df = fact_points_df \
    .withColumn("P1_Set_Win_Add", F.when((F.col("New_Set_Start") == 1) & (F.col("Prev_PointWinner") == 1), 1).otherwise(0)) \
    .withColumn("P2_Set_Win_Add", F.when((F.col("New_Set_Start") == 1) & (F.col("Prev_PointWinner") == 2), 1).otherwise(0))

# Cumulative sum over match to get number of sets won *before* this point
fact_points_df = fact_points_df \
    .withColumn("P1SetsWon_Pre", F.sum("P1_Set_Win_Add").over(match_point_window)) \
    .withColumn("P2SetsWon_Pre", F.sum("P2_Set_Win_Add").over(match_point_window))

# Optional: Drop helper columns if no longer needed
fact_points_df = fact_points_df.drop("Prev_SetNo", "Prev_PointWinner", "New_Set_Start", "P1_Set_Win_Add", "P2_Set_Win_Add")

fact_points_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("fact_points")


# COMMAND ----------

# MAGIC %run /Workspace/Users/cwr321@gmail.com/point_importance_simulations
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dim_player LIMIT 1000;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM fact_points WHERE match_key between 0 and 99;

# COMMAND ----------

from pyspark.sql.functions import when, col

score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}

fact_points = fact_points.withColumn(
    "P1Score_Pre_Int",
    when(col("P1Score_Pre") == '0', 0)
    .when(col("P1Score_Pre") == '15', 1)
    .when(col("P1Score_Pre") == '30', 2)
    .when(col("P1Score_Pre") == '40', 3)
    .when(col("P1Score_Pre") == 'AD', 4)
    .otherwise(0)  # fallback if unknown value
)

fact_points = fact_points.withColumn(
    "P2Score_Pre_Int",
    when(col("P2Score_Pre") == '0', 0)
    .when(col("P2Score_Pre") == '15', 1)
    .when(col("P2Score_Pre") == '30', 2)
    .when(col("P2Score_Pre") == '40', 3)
    .when(col("P2Score_Pre") == 'AD', 4)
    .otherwise(0)
)


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import when, col

import pandas as pd
import numpy as np
import re
import time


# --- Assume your numba functions and helpers are imported here ---
# simulate_game, simulate_tiebreak, simulate_set, simulate_match,
# monte_carlo_win_prob_from_state, compute_point_importance, tennis_score_to_int

table_name = "point_importance"
first_chunk = True

# --- 2. Load dim_player once ---
dim_player = spark.table("dim_player").select("player_id", "serve_point_win_pct", "return_point_win_pct")

# --- 1. Load fact_points filtered on match_key 100-199 ---
fact_points = spark.table("fact_points")



for start in range(1800, 1810, 10):
#for start in range(0, 10513, 10):

    end = start + 9
    print(f"Processing chunk {start}-{end}")
    t0 = time.time()

    fact_points_chunk = spark.table("fact_points").filter(
        (F.col("match_key") >= start) & (F.col("match_key") <= end)
    )
    
    dim_match = spark.table("dim_match").select("match_key", "tour")
    fact_points_chunk = fact_points_chunk.join(dim_match, on="match_key", how="left")

    fact_points_chunk = fact_points_chunk.withColumn(
        "best_of_5",
        when(F.col("tour").isin("ATP", "Both"), True).otherwise(False)
    )



    # --- Define mapping as Spark map literal ---
    score_map_expr = F.create_map(
        F.lit('0'), F.lit(0),
        F.lit('15'), F.lit(1),
        F.lit('30'), F.lit(2),
        F.lit('40'), F.lit(3),
        F.lit('AD'), F.lit(4)
    )

    fact_points_chunk = fact_points_chunk.withColumn(
        "P1Score_Pre_Int",
        F.coalesce(score_map_expr[F.col("P1Score_Pre")], F.lit(None))
    )

    fact_points_chunk = fact_points_chunk.withColumn(
        "P2Score_Pre_Int",
        F.coalesce(score_map_expr.getItem(F.col("P2Score_Pre")), F.lit(None))
    )

    # --- 3. Join to get serve/return stats for server and returner ---
    fact_points_chunk = fact_points_chunk \
        .join(dim_player.alias("p1"), fact_points_chunk["server_id"] == F.col("p1.player_id"), "left") \
        .join(dim_player.alias("p2"), fact_points_chunk["returner_id"] == F.col("p2.player_id"), "left") \
        .select(
            fact_points_chunk["*"],
            F.col("p1.serve_point_win_pct").alias("p1_serve_point_win_pct"),
            F.col("p1.return_point_win_pct").alias("p1_return_point_win_pct"),
            F.col("p2.serve_point_win_pct").alias("p2_serve_point_win_pct"),
            F.col("p2.return_point_win_pct").alias("p2_return_point_win_pct"),
        )

    # --- Cast all relevant columns to integer or float types ---
    fact_points_chunk = fact_points_chunk \
        .withColumn("P1Score_Pre_Int", F.col("P1Score_Pre_Int").cast("int")) \
        .withColumn("P2Score_Pre_Int", F.col("P2Score_Pre_Int").cast("int")) \
        .withColumn("P1SetsWon_Pre", F.col("P1SetsWon_Pre").cast("int")) \
        .withColumn("P2SetsWon_Pre", F.col("P2SetsWon_Pre").cast("int")) \
        .withColumn("P1GamesWon_Pre", F.col("P1GamesWon_Pre").cast("int")) \
        .withColumn("P2GamesWon_Pre", F.col("P2GamesWon_Pre").cast("int")) \
        .withColumn("PointServer", F.col("PointServer").cast("int")) \
        .withColumn("p1_serve_point_win_pct", F.col("p1_serve_point_win_pct").cast("double")) \
        .withColumn("p1_return_point_win_pct", F.col("p1_return_point_win_pct").cast("double")) \
        .withColumn("p2_serve_point_win_pct", F.col("p2_serve_point_win_pct").cast("double")) \
        .withColumn("p2_return_point_win_pct", F.col("p2_return_point_win_pct").cast("double"))

    count = fact_points_chunk.count()
    print(f"Chunk {start}-{end} row count: {count}")
    fact_points_chunk.select("point_id", "match_key").show(10, truncate=False)

    # --- 4. Define schema for UDF output ---
    schema = StructType([
        StructField("point_id", IntegerType()),
        StructField("match_key", IntegerType()),
        StructField("p1_win_prob_before", DoubleType()),
        StructField("p1_win_prob_if_p1_wins", DoubleType()),
        StructField("p1_win_prob_if_p2_wins", DoubleType()),
        StructField("importance", DoubleType()),
        StructField("error", StringType()),  # add this line!
    ])

    def row_simulation(pdf):
        results = []
        errors = []

        for idx, row in pdf.iterrows():
            print(f"Processing point_id={row['point_id']} match_key={row['match_key']}")  # Debug print
            try:
                # --- Determine if this is a tiebreak ---
                is_tiebreak = (int(row['P1GamesWon_Pre']) == 6 and int(row['P2GamesWon_Pre']) == 6)

                # --- Map numeric scores to tennis score strings ---
                if row['PointServer'] == 1:
                    p1_raw = int(row['P1Score_Pre_Int'])
                    p2_raw = int(row['P2Score_Pre_Int'])
                else:
                    p1_raw = int(row['P2Score_Pre_Int'])
                    p2_raw = int(row['P1Score_Pre_Int'])

                # --- Convert tennis score strings to int BEFORE simulation ---
                p1_points = tennis_score_to_int(p1_raw, is_tiebreak)
                p2_points = tennis_score_to_int(p2_raw, is_tiebreak)

                # --- Extract serve/return stats and sets/games won ---
                if row['PointServer'] == 1:
                    server1_wp = row['p1_serve_point_win_pct']
                    returner2_wp = row['p2_return_point_win_pct']
                    server2_wp = row['p2_serve_point_win_pct']
                    returner1_wp = row['p1_return_point_win_pct']
                    p1_sets = int(row['P1SetsWon_Pre'])
                    p2_sets = int(row['P2SetsWon_Pre'])
                    p1_games = int(row['P1GamesWon_Pre'])
                    p2_games = int(row['P2GamesWon_Pre'])
                else:
                    server1_wp = row['p2_serve_point_win_pct']
                    returner2_wp = row['p1_return_point_win_pct']
                    server2_wp = row['p1_serve_point_win_pct']
                    returner1_wp = row['p2_return_point_win_pct']
                    p1_sets = int(row['P2SetsWon_Pre'])
                    p2_sets = int(row['P1SetsWon_Pre'])
                    p1_games = int(row['P1GamesWon_Pre'])
                    p2_games = int(row['P2GamesWon_Pre'])

                # --- Call the compute_point_importance with clean ints ---
                is_best_of_5 = bool(row.get("best_of_5", False))  # Handle possible None
                best_of = 5 if is_best_of_5 else 3

                res = compute_point_importance(
                    server1_wp, returner2_wp, server2_wp, returner1_wp,
                    p1_sets, p2_sets,
                    p1_games, p2_games,
                    p1_points, p2_points,
                    row['PointServer'],
                    True,
                    best_of,
                    n_simulations=50
                )

                res['point_id'] = row['point_id']
                res['match_key'] = row['match_key']
                errors.append("")  # No error
                results.append(res)

            except Exception as e:
                print(f"❌ Error in point_id={row.get('point_id')}, match_key={row.get('match_key')}: {e}")
                errors.append(str(e))
                results.append({
                    'point_id': row['point_id'],
                    'match_key': row['match_key'],
                    'p1_win_prob_before': None,
                    'p1_win_prob_if_p1_wins': None,
                    'p1_win_prob_if_p2_wins': None,
                    'importance': None
                })

        df_out = pd.DataFrame(results)
        df_out['error'] = errors
        return df_out

    # --- 6. Define GROUPED_MAP pandas UDF ---
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def simulate_importance_udf(pdf: pd.DataFrame) -> pd.DataFrame:
        return row_simulation(pdf)

    # --- 7. Add dummy column for grouping all data together ---
    fact_points_dummy = fact_points_chunk.withColumn("dummy", F.lit(1))

    # --- 8. Apply UDF by grouping on dummy (runs on all data as one group) ---
    result_df = fact_points_dummy.groupBy("dummy").apply(simulate_importance_udf).drop("dummy")

    # --- 9. Show or save results ---
    if first_chunk:
        # On first chunk: create the table by overwriting if exists
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        result_df.write.mode("overwrite").format("delta").saveAsTable(table_name)
        first_chunk = False
    else:
        # For later chunks: append to the existing table
        result_df.write.format("delta").mode("append").saveAsTable(table_name)

    print(f"✅ Saved chunk {start}-{end} in {round(time.time() - t0, 2)} seconds")


# COMMAND ----------

fact_points_chunk.select("point_id", "match_key",'importance').show(1000, truncate=False)


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM point_importance limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from point_importance 

# COMMAND ----------

result_df.count()