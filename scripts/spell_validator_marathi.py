import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# -----------------------------
# CONFIG
# -----------------------------
FILE = "../data/Master_names_marathi_sample.xlsx"  # Update to your Marathi dataset file
COL = "old_value"
OUTPUT = "../results/final_clusters_with_suggestions_marathi.xlsx"

# -----------------------------
# CUSTOM PHONETIC CODE
# -----------------------------
def desi_phonetic_code(name):
    n = name.upper().strip()

    # Collapse repeated vowels
    n = re.sub(r"A+", "A", n)
    n = re.sub(r"E+", "E", n)
    n = re.sub(r"I+", "I", n)
    n = re.sub(r"O+", "O", n)
    n = re.sub(r"U+", "U", n)

    # Common vowel combinations → A
    n = re.sub(r"AI|AY|AU|AW", "A", n)

    # Keep first vowel, drop other vowels
    if len(n) > 0:
        n = n[0] + re.sub(r"[AEIOU]", "", n[1:])

    # Consonant normalization
    repl = {
        "SH": "S", "CH": "C", "DH": "D", "TH": "D",
        "BH": "B", "PH": "F", "GH": "G", "Z": "J"
    }
    for k, v in repl.items():
        n = n.replace(k, v)

    # Common Indian surname tails
    n = re.sub(r"RAO$", "R", n)
    n = re.sub(r"KAR$", "KR", n)
    n = re.sub(r"WAR$", "WR", n)
    n = re.sub(r"VAR$", "WR", n)

    # Remove repeated consonants
    n = re.sub(r"(.)\1+", r"\1", n)

    return n

# -----------------------------
# PICK CANONICAL SPELLING
# -----------------------------
def canonical_by_shortest(names):
    """Choose the shortest spelling → simplest & least noisy."""
    names = sorted(list(set(names)), key=len)
    return names[0]

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel(FILE)
df[COL] = df[COL].astype(str).str.strip()  # Strip spaces, but no upper() for Devanagari

# Extract Romanized transliteration from linked_value (take first value)
def extract_roman(linked_str):
    if pd.isna(linked_str) or linked_str == '':
        return ''
    try:
        linked_list = json.loads(linked_str)
        if linked_list and len(linked_list) > 0 and 'value' in linked_list[0]:
            return linked_list[0]['value'].upper().strip()
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return ''

df["roman"] = df["linked_value"].apply(extract_roman)

# Use roman for processing (similarity/clustering)
names_roman = df["roman"].tolist()

# -----------------------------
# SOUND CLUSTERING (Embeddings on Romanized)
# -----------------------------
embedder = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

emb = embedder.encode(names_roman)
emb = normalize(emb)

sound_cluster = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=1.25,
    linkage="ward"
).fit(emb)

df["sound_cluster_id"] = sound_cluster.labels_

# -----------------------------
# PHONETIC CLUSTERING (on Romanized)
# -----------------------------
df["phonetic_base"] = df["roman"].apply(desi_phonetic_code)
df["phonetic_cluster_id"] = df["phonetic_base"]

# -----------------------------
# BUILD SUGGESTIONS (Canonical in Marathi script)
# -----------------------------
sound_suggestion = {}
phonetic_suggestion = {}

# Suggestion based on SOUND cluster (map back to Marathi old_value)
for cid, group in df.groupby("sound_cluster_id"):
    gnames_mar = group[COL].tolist()  # Use Marathi names for canonical choice
    canon_mar = canonical_by_shortest(gnames_mar)
    for n_mar in gnames_mar:
        sound_suggestion[n_mar] = canon_mar

# Suggestion based on PHONETIC cluster
for cid, group in df.groupby("phonetic_cluster_id"):
    gnames_mar = group[COL].tolist()  # Use Marathi names for canonical choice
    canon_mar = canonical_by_shortest(gnames_mar)
    for n_mar in gnames_mar:
        phonetic_suggestion[n_mar] = canon_mar

df["suggestion_sound"] = df[COL].map(sound_suggestion)
df["suggestion_phonetic"] = df[COL].map(phonetic_suggestion)

# -----------------------------
# USER CORRECTION FLAG
# 1 = corrected
# 0 = no change
# -----------------------------
df["user_correction_flag"] = df.apply(
    lambda row: 1 if (row["suggestion_sound"] != row[COL] or 
                      row["suggestion_phonetic"] != row[COL]) else 0,
    axis=1
)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_excel(OUTPUT, index=False)
print("\n✔ DONE — Output saved to:", OUTPUT)

# gaiekwad
# gaikwad
# gayakwad
# gwad
# gaykwad
# grayward lol 
# its not possible to spot out mistakes like that
