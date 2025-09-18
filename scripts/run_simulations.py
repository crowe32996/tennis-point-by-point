import sys
import os

# Add the project root (one level above scripts/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17.0.16+8"
os.environ["PYSPARK_PYTHON"] = r"C:\Users\peppe\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\peppe\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["HADOOP_HOME"] = r"C:\hadoop\hadoop-3.3.6"
os.environ["PATH"] = r"C:\hadoop\hadoop-3.3.6\bin;" + os.environ["PATH"]


import pandas as pd
import duckdb
import simulations.point_importance_simulation as pis
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, when, struct, create_map, lit, lower
from pyspark.sql.types import StructType, StructField, DoubleType
from itertools import chain

INPUT_FILE = "data/processed/merged_tennis_data.csv"
OUTPUT_FILE = "outputs/all_points_with_importance.csv"
PARQUET_FILE = r"C:\Users\peppe\OneDrive\Desktop\Charlie\Data_Projects\tennis-point-by-point\outputs\all_points_with_importance.parquet"
#CHUNK_SIZE = 10000
N_SIMULATIONS = 1000
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

    # Read CSV as Spark DataFrame
    df_spark = spark.read.csv(INPUT_FILE, header=True, inferSchema=True)

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

    #df_spark = df_spark.limit(100)  # <--- only 100 rows, full 1000 sims will run on them

    # Add column (renamed to points_stake for clarity)
    df_spark = df_spark.withColumn("points_stake", mapping_expr[col("round")])

    # --- Add repartition and optional caching ---
    df_spark = df_spark.repartition(16)  # split into 16 parallel tasks (adjust to number of cores)
    df_spark.cache()  # keeps it in memory if used multiple times

    #chunk_iter = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)
    con = duckdb.connect("outputs/sim_results.duckdb")

    # Ask whether to rerun simulations
    rerun_sim = prompt_yes_no("Recompute simulation from scratch?")

    if rerun_sim:
        print("Running full simulation with Spark and overwriting DuckDB table...")

        # --- Define the schema of the UDF output
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

        df_spark.write.mode("overwrite").parquet(PARQUET_FILE)

        con.execute(f"""
            CREATE OR REPLACE TABLE {TABLE_NAME}
            AS SELECT * FROM parquet_scan('{PARQUET_FILE}/*.parquet');
        """)

        # Save full CSV as a single file for easy inspection
        df_spark.coalesce(1).write.csv(OUTPUT_FILE, header=True, mode="overwrite")
        print(f"Done! Full results saved to {OUTPUT_FILE}")

        con.close()
        spark.stop()

    else:
        print("Skipping simulation – using existing importance_results table.")

if __name__ == "__main__":
    main()