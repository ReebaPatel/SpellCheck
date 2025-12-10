import pandas as pd
from metaphone import doublemetaphone
from collections import defaultdict

# ----------------------------
# INPUT FILE CONFIG
# ----------------------------
INPUT_FILE = "../data/all_surnames.xlsx"    # your mixed surname list
COLUMN_NAME = "Surnames"             # change if needed
OUTPUT_FILE = "sound_clusters.xlsx"

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_excel(INPUT_FILE)
df[COLUMN_NAME] = df[COLUMN_NAME].astype(str).str.strip().str.upper()

# ----------------------------
# GROUP NAMES BY PHONETIC CODE
# ----------------------------
clusters = defaultdict(list)

for name in df[COLUMN_NAME]:
    code1, code2 = doublemetaphone(name)

    # Prefer primary code
    key = code1 if code1 else code2

    clusters[key].append(name)

# ----------------------------
# CONVERT CLUSTERS → EXCEL-FRIENDLY FORMAT
# ----------------------------
cluster_rows = []

for code, names in clusters.items():
    cluster_rows.append({
        "metaphone_code": code,
        "count": len(names),
        "names": ", ".join(sorted(set(names)))
    })

cluster_df = pd.DataFrame(cluster_rows).sort_values(by="count", ascending=False)
cluster_df.to_excel(OUTPUT_FILE, index=False)

print("DONE ✔ Clusters saved to", OUTPUT_FILE)
