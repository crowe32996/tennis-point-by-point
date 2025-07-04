import pandas as pd
import duckdb
from simulations.point_importance_simulation import compute_importance_for_df

INPUT_FILE = "data/processed/merged_tennis_data.csv"
OUTPUT_FILE = "outputs/all_points_with_importance.csv"
CHUNK_SIZE = 10000
N_SIMULATIONS = 1000
TABLE_NAME = "importance_results"

def filter_valid_pointnumber(df, col='PointNumber'):
    # Keep only rows where PointNumber can be converted to int (digits only)
    return df[df[col].astype(str).str.match(r'^\d+$')]

def main():
    chunk_iter = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)
    con = duckdb.connect("outputs/sim_results.duckdb")
    first_chunk = True

    for i, chunk in enumerate(chunk_iter):
        print(f"Processing chunk {i + 1} with {len(chunk)} rows...")

        # Filter out rows where PointNumber is not all digits
        chunk = filter_valid_pointnumber(chunk, 'PointNumber')

        # Run importance simulation
        result = compute_importance_for_df(chunk.copy(), n_simulations=N_SIMULATIONS)

        # Drop ElapsedTime if it exists
        if 'ElapsedTime' in result.columns:
            result = result.drop(columns=['ElapsedTime'])

        if first_chunk:
            # Create table from schema of empty DataFrame
            con.execute(f"""
                CREATE TABLE {TABLE_NAME} AS 
                SELECT * FROM result WHERE 0=1
            """)

            con.append(TABLE_NAME, result)
            first_chunk = False
        else:
            con.append(TABLE_NAME, result)

    print("All chunks processed and written to DuckDB.")

    df_full = con.execute(f"SELECT * FROM {TABLE_NAME}").fetchdf()
    df_full.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Full results saved to {OUTPUT_FILE}")

    con.close()

if __name__ == "__main__":
    main()