import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer

INPUT_FILE = "Master_names_sample.xlsx"
EMB_FILE = "embeddings/name_embeddings.npy"
LIST_FILE = "embeddings/name_list.json"


print("Loading dataset...")
df = pd.read_excel(INPUT_FILE)
names = df["old_value"].astype(str).str.upper().unique().tolist()

print("Loaded:", len(names), "unique names")

print("Loading MPNet model...")
model = SentenceTransformer("paraphrase-mpnet-base-v2")

print("Encoding names...")
emb = model.encode(names, batch_size=64, show_progress_bar=True)

print("Saving embeddings...")
np.save(EMB_FILE, emb)

with open(LIST_FILE, "w", encoding="utf-8") as f:
    json.dump(names, f, ensure_ascii=False, indent=2)

print("DONE — embeddings saved!")
