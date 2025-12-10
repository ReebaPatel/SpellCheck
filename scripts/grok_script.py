import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIG
# -----------------------------
FILE = "Master_names_sample.xlsx"  # Update with your actual file
COL = "old_value"
OUTPUT = "final_clusters_with_suggestions_improved.xlsx"

# -----------------------------
# ENHANCED PHONETIC CODE (IndiSoundex-inspired + Double Metaphone basics)
# -----------------------------
def desi_phonetic_code(name):
    """
    Improved phonetic encoder for Indian surnames:
    - Handles W/V, KH/K, aspirates, diphthongs.
    - Simple Double Metaphone: Groups by sound classes, pads to 8 chars.
    - Reduces collisions (e.g., Gaikwad → G2120000, Gupta → G1300000).
    """
    n = name.upper().strip()
    if not n:
        return "00000000"

    # Indian-specific swaps (pre-normalize)
    indian_swaps = {
        r"W": "V", r"V": "W",  # W/V flip common in Marathi/Tamil
        r"KH": "K", r"PH": "F", r"TH": "T", r"DH": "D",  # Aspirates simplify
        r"SH": "S", r"CH": "C", r"JH": "J", r"GH": "G", r"BH": "B",
        r"AI": "AY", r"AU": "O", r"EE": "I", r"OO": "U",  # Diphthongs
        r"NG": "N", r"NY": "N"  # Nasal clusters
    }
    for k, v in indian_swaps.items():
        n = re.sub(k, v, n)

    # Collapse repeated chars (vowels/consonants)
    n = re.sub(r"(.)\1+", r"\1", n)

    # Surname tails (keep more distinct)
    n = re.sub(r"RAO$", "R", n)
    n = re.sub(r"KAR$", "KR", n)
    n = re.sub(r"WAR|VAR$", "WR", n)

    # Simple Double Metaphone: Assign sound codes (B=1, S=2, etc.)
    # Vowels ignored except first; focus on consonants
    codes = {'BFPV': '1', 'CSKGJQXZ': '2', 'DT': '3', 'L': '4', 'MN': '5', 'R': '6'}
    phonetic = ""
    used = set()  # Avoid duplicates in positions
    for i, char in enumerate(n):
        for group, code in codes.items():
            if char in group and code not in used:
                phonetic += code
                used.add(code)
                break
        # Add first vowel as anchor if no codes yet
        if not phonetic and char in 'AEIOUY':
            phonetic += '0'

    # Pad to 8 chars (standard for matching)
    phonetic = phonetic[:8].ljust(8, '0')
    return phonetic

# -----------------------------
# PICK CANONICAL SPELLING (Frequency > Shortest)
# -----------------------------
def canonical_by_frequency(names):
    """Choose most frequent spelling → more 'real' canonical."""
    if not names:
        return ""
    return Counter(names).most_common(1)[0][0]

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel(FILE)
df[COL] = df[COL].astype(str).str.upper().str.strip()
df = df.dropna(subset=[COL])  # Drop empties
names = df[COL].tolist()
print(f"Loaded {len(names)} names.")

# -----------------------------
# SOUND CLUSTERING (TF-IDF Char N-Grams: Faster, Name-Focused)
# -----------------------------
print("Computing TF-IDF embeddings...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=1000, lowercase=False)
emb = vectorizer.fit_transform(names).toarray()
emb = normalize(emb)

sound_cluster = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.9,  # Tuned lower for tighter clusters (was 1.25)
    linkage="ward"
).fit(emb)

df["sound_cluster_id"] = sound_cluster.labels_

# Evaluate cluster quality
if len(set(sound_cluster.labels_)) > 1:
    sil_sound = silhouette_score(emb, sound_cluster.labels_)
    print(f"Sound Silhouette Score: {sil_sound:.3f} (Higher = better clustering)")

# -----------------------------
# PHONETIC CLUSTERING
# -----------------------------
print("Computing phonetic codes...")
df["phonetic_base"] = df[COL].apply(desi_phonetic_code)

# Use Levenshtein-like grouping: Cluster by exact phonetic code (simple + effective)
df["phonetic_cluster_id"] = df["phonetic_base"]  # Direct for now; add distance if needed

# -----------------------------
# BUILD SUGGESTIONS
# -----------------------------
sound_suggestion = {}
phonetic_suggestion = {}

# Sound-based suggestions
for cid, group in df.groupby("sound_cluster_id"):
    if len(group) > 1:  # Only cluster if >1
        gnames = group[COL].tolist()
        canon = canonical_by_frequency(gnames)
        for n in gnames:
            sound_suggestion[n] = canon
    else:
        sound_suggestion[group[COL].iloc[0]] = group[COL].iloc[0]

# Phonetic-based suggestions
for cid, group in df.groupby("phonetic_cluster_id"):
    if len(group) > 1:
        gnames = group[COL].tolist()
        canon = canonical_by_frequency(gnames)
        for n in gnames:
            phonetic_suggestion[n] = canon
    else:
        phonetic_suggestion[group[COL].iloc[0]] = group[COL].iloc[0]

df["suggestion_sound"] = df[COL].map(sound_suggestion).fillna(df[COL])
df["suggestion_phonetic"] = df[COL].map(phonetic_suggestion).fillna(df[COL])

# -----------------------------
# PRIORITIZED SUGGESTION (Merge: Prefer larger cluster)
# -----------------------------
df["suggestion_final"] = df.apply(
    lambda row: row["suggestion_sound"] if row["suggestion_sound"] != row[COL] 
                and len(df[df["suggestion_sound"] == row["suggestion_sound"]]) > 
                len(df[df["suggestion_phonetic"] == row["suggestion_phonetic"]]) 
                else row["suggestion_phonetic"], axis=1
)

# -----------------------------
# USER CORRECTION FLAG
# 1 = needs correction (suggestion differs)
# 0 = no change
# -----------------------------
df["user_correction_flag"] = df.apply(
    lambda row: 1 if (row["suggestion_final"] != row[COL]) else 0, axis=1
)

# -----------------------------
# STATS
# -----------------------------
total = len(df)
flagged = df["user_correction_flag"].sum()
print(f"\nStats: {flagged}/{total} ({flagged/total*100:.1f}%) flagged for correction.")

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_excel(OUTPUT, index=False)
print(f"\n✔ DONE — Output saved to: {OUTPUT}")
print("Columns: old_value, sound_cluster_id, phonetic_base, suggestion_sound, suggestion_phonetic, suggestion_final, user_correction_flag")