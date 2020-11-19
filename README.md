# ent2id



A lightweight program to disambiguate entity names and convert it to IDs.

The method is based on a *supervised n-gram* Word2Vec model, where the input layer is the n-gram of entity names and the output layer is the ID of the ID.
The embedding dimention is infinate, as the aim is predictions instead of embedding (in case of infinate dimensions, analytical solution can be easily computed.)

## Usage

```python
model = ent2id.Ent2Id()
model.fit(entity_name_list, id_list)
ids, score = model.predict(test_entity_name_list)
```
- `entity_name_list`: List of entity names
- `id_list` : id_list[i] indicates the id of the entity name given by `entity_name_list[i]`
- `test_entity_name_list`: the model predicts an entity label for each `test_entity_name_list[i]`.

### For large dataset

```python
import pandas as pd
import ent2id

entity_table_name = "example/sampled-grid.csv"

model = ent2id.Ent2Id()
for df in pd.read_csv(entity_table_name, chunksize = 500):
    model.partial_fit(df["Name"].values, df["ID"].values)

for df in pd.read_csv(entity_table_name, chunksize = 500):
    ids, score = model.predict(df.Name.values)
    print("true positive rate %f" % np.sum(ids == df.ID)/df.shape[0] )
```
