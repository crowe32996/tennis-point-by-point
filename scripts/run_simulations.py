import sys
import os
from pathlib import Path

# Add the project root (one level above scripts/) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Configure Spark environment from env vars (set these in your .env or shell profile)
# Required: JAVA_HOME, HADOOP_HOME (Windows only)
# Optional: PYSPARK_PYTHON, PYSPARK_DRIVER_PYTHON
if "JAVA_HOME" not in os.environ:
    print("Warning: JAVA_HOME not set. Please set it to your JDK installation path.")

# Auto-detect Python executable for PySpark if not set
if "PYSPARK_PYTHON" not in os.environ:
    os.environ["PYSPARK_PYTHON"] = sys.executable
if "PYSPARK_DRIVER_PYTHON" not in os.environ:
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# On Windows, HADOOP_HOME may be needed for winutils
if sys.platform == "win32" and "HADOOP_HOME" in os.environ:
    hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
    if hadoop_bin not in os.environ["PATH"]:
        os.environ["PATH"] = hadoop_bin + ";" + os.environ["PATH"]

import pandas as pd
import duckdb
import simulations.point_importance_simulation as pis
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, when, struct, create_map, lit, lower
from pyspark.sql.types import StructType, StructField, DoubleType
from itertools import chain
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import time

# Map tennis score strings to numeric points
score_map = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}

def score_to_int(score_str):
    return score_map.get(str(score_str), 0)

score_udf = udf(score_to_int, IntegerType())

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "merged_tennis_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "all_points_with_importance.csv"
PARQUET_FILE = PROJECT_ROOT / "outputs" / "all_points_with_importance.parquet"
DUCKDB_FILE = PROJECT_ROOT / "outputs" / "sim_results.duckdb"
N_SIMULATIONS = 500
TABLE_NAME = "importance_results"

def prompt_yes_no(question):
    while True:
        choice = input(f"{question} (y/n): ").strip().lower()
        if choice in ['y', 'n']:
            return choice == 'y'
        print("Please respond with 'y' or 'n'.")

def main():
    spark = SparkSession.builder \
    .appName("TennisPointImportance") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()
    
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # Read CSV as Spark DataFrame
    df_spark = spark.read.csv(str(INPUT_FILE), header=True, inferSchema=True)

    # keep only rows where PointNumber is numeric
    df_spark = df_spark.filter(col("PointNumber").rlike("^[0-9]+$"))

    # Keep only rows where PointNumber is numeric
    df_spark = df_spark.filter(col("PointNumber").cast("int").isNotNull())

    df_spark = df_spark.withColumn(
        "best_of_5",
        col("best_of_5").cast("int")
    )

    round_points_map = {
        1: 10,    # R128
        2: 45,    # R64
        3: 90,    # R32
        4: 180,   # R16
        5: 360,   # QF
        6: 720,   # SF
        7: 1200,  # F
        8: 2000   # Winner
    }

    # Convert dict into Spark map expression
    mapping_expr = create_map([lit(x) for x in chain(*round_points_map.items())])

    # df_spark = df_spark.limit(500000)  # <--- only 100 rows, full 1000 sims will run on them

    # Add column (renamed to points_stake for clarity)
    df_spark = df_spark.withColumn("points_stake", mapping_expr[col("round")])

    # --- Add repartition and optional caching ---
    df_spark = df_spark.repartition(16)  # split into 16 parallel tasks (adjust to number of cores)
    df_spark.cache()  # keeps it in memory if used multiple times

    con = duckdb.connect(str(DUCKDB_FILE))

    # Ask whether to rerun simulations
    rerun_sim = prompt_yes_no("Recompute simulation from scratch?")

    if rerun_sim:
        print("Running full simulation with Spark and overwriting DuckDB table...")
        start_time = time.time()

        # --- Convert score strings to numeric points for the optimized UDF ---
        df_spark = df_spark.withColumn("p1_points", score_udf("P1Score_Pre")) \
                        .withColumn("p2_points", score_udf("P2Score_Pre"))

        # --- Define the schema of the UDF output ---
        schema = StructType([
            StructField("p1_win_prob_before", DoubleType(), True),
            StructField("p1_win_prob_if_p1_wins", DoubleType(), True),
            StructField("p1_win_prob_if_p2_wins", DoubleType(), True),
            StructField("importance", DoubleType(), True)
        ])

        # --- Define Pandas UDF
        @pandas_udf(schema)
        def importance_udf(pdf: pd.DataFrame) -> pd.DataFrame:
            return pis.importance_batch_fn(pdf, n_simulations=N_SIMULATIONS)

        df_spark = df_spark.withColumn(
            "importance_results",
            importance_udf(struct(*df_spark.columns))
        )

        # --- Explode nested struct into separate columns
        df_spark = df_spark.select("*", "importance_results.*").drop("importance_results")

        df_spark = df_spark.withColumn(
            "p1_wp_delta",
            when(
                col("PointWinner") == 1,
                col("p1_win_prob_if_p1_wins") - col("p1_win_prob_before")
            ).when(
                col("PointWinner") == 2,
                col("p1_win_prob_if_p2_wins") - col("p1_win_prob_before")
            )
        ).withColumn(
            "p2_wp_delta",
            -col("p1_wp_delta")
        )

        # Drop ElapsedTime if it exists
        if "ElapsedTime" in df_spark.columns:
            df_spark = df_spark.drop("ElapsedTime")

        df_spark = df_spark.withColumn(
            "match_winner_prob_before",
            when(col("match_winner") == col("player1"), col("p1_win_prob_before"))
            .otherwise(col("p1_win_prob_if_p2_wins"))
        )

        df_spark.write.mode("overwrite").parquet(str(PARQUET_FILE))

        con.execute(f"""
            CREATE OR REPLACE TABLE {TABLE_NAME}
            AS SELECT * FROM parquet_scan('{PARQUET_FILE}/*.parquet');
        """)
        df_spark = df_spark.sort(["match_id", "SetNo", "PointNumber"])

        # Save full CSV as a single file for easy inspection
        df_spark.write.csv(str(OUTPUT_FILE), header=True, mode="overwrite")
        print(f"Done! Full results saved to {OUTPUT_FILE}")

        con.close()

        end_time = time.time()
        elapsed = end_time - start_time
        num_rows = df_spark.count()  # actual number of rows processed
        total_sims = num_rows * N_SIMULATIONS

        print(f"Simulation completed for {num_rows} rows with {N_SIMULATIONS} simulations per row.")
        print(f"Total point probability simulations: ~{total_sims}")
        print(f"Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"Average time per simulation: {elapsed/total_sims:.6f} seconds")

        spark.stop()

    else:
        print("Skipping simulation – using existing importance_results table.")

if __name__ == "__main__":
    main()