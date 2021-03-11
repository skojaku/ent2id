# %%
import numpy as np
import pandas as pd
from scipy import sparse
import re
sys.path.insert(0, "..")
import ent2id

entity_table_name = "sample-grid.csv"

model = ent2id.Ent2Id(aggregate_duplicates = False)
for df in pd.read_csv("sample-grid.csv", chunksize=5000):
    model.partial_fit(df["Name"].values, df["ID"].values)

for df in pd.read_csv("sample-grid.csv", chunksize=5000):
    ids, score = model.predict(df.Name.values)
    print("Precision %f" % (np.sum(ids == df.ID) / df.shape[0]) )

# %%
