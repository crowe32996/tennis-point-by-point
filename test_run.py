import pandas as pd
pdf = pd.read_csv("data/processed/merged_tennis_data.csv").head(100)
import simulations.point_importance_simulation as pis

res = pis.importance_batch_fn(pdf, n_simulations=50)
print(res.head())
