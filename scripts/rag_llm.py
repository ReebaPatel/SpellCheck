import pandas as pd
import numpy as np
import json
import subprocess
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
RAW_FILE = "Master_names_sample.xlsx"
NAME_LIST_FILE = "embeddings/name_list.json"
EMBED_FILE = "embeddings/name_embeddings.npy"

OUTPUT_FULL = "validated_output.xlsx"
OUTPUT_MISTAKES = "validated_mistakes_only.xlsx"

MODEL_NAME = "llama3.2:1b"  # MUST MATCH ollama list


# -------------------------
# JSON CLEANER (bulletproof)
# -------------------------
def extract_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("JSON not found")
        cleaned = text[start: end + 1]
        return json.loads(cleaned)
    except:
        return {
            "valid": False,
            "suggestion": "",
            "reason": "Invalid JSON"
        }


# -------------------------
# LOAD EMBEDDING MODEL
# -------------------------
print("Loading MPNet embedding model...")
embedder = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")


# -------------------------
# LLM CALL
# -------------------------
def ask_llama(name, nearest_names):
    nearest_str = ", ".join(nearest_names)

    prompt = f"""
You are an Indian name validator.

Decide if "{name}" is a valid Indian name or surname.

Similar names from user dataset:
{nearest_str}

Rules:
- Consider Indian phonetics.
- Consider all Indian regional variations.
- Do NOT assume western spellings.
- VALID = realistic Indian spelling.
- If INVALID → suggest the MOST LIKELY correct spelling.
- Suggest ONLY ONE best spelling.

Output JSON ONLY:
{{
 "valid": true/false,
 "suggestion": "string",
 "reason": "short explanation"
}}
"""

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            timeout=15
        )

        raw = result.stdout.decode().strip()
        return extract_json(raw)

    except subprocess.TimeoutExpired:
        return {"valid": False, "suggestion": "", "reason": "timeout"}

    except:
        return {"valid": False, "suggestion": "", "reason": "LLM error"}


# -------------------------
# LOAD DATABASE
# -------------------------
print("Loading database embeddings...")

with open(NAME_LIST_FILE, "r", encoding="utf-8") as f:
    name_list = json.load(f)

embeddings = np.load(EMBED_FILE)
name_to_index = {name_list[i]: i for i in range(len(name_list))}

print(f"Loaded {len(name_list)} known names.")


# -------------------------
# LOAD RAW FILE
# -------------------------
print("Loading raw dataset...")
df = pd.read_excel(RAW_FILE)

# 🔥🔥 RANDOM SAMPLE OF 30 NAMES (NEW)
df = df.sample(30, random_state=42)

df["old_value"] = df["old_value"].astype(str).str.strip().str.upper()


# -------------------------
# NEAREST NAME FINDER
# -------------------------
def get_nearest_names(query, top_k=3):
    query = query.upper()

    if query in name_to_index:
        idx = name_to_index[query]
        q_emb = embeddings[idx].reshape(1, -1)
    else:
        q_emb = embedder.encode([query])

    sims = cosine_similarity(q_emb, embeddings)[0]
    nearest_ids = sims.argsort()[::-1][:top_k]

    return [name_list[i] for i in nearest_ids]


# -------------------------
# VALIDATION LOOP
# -------------------------
print("\nStarting LLM validation...\n")

rows = []

for name in tqdm(df["old_value"], desc="Validating names"):
    nearest = get_nearest_names(name, top_k=3)
    out = ask_llama(name, nearest)

    flag = 0 if out["valid"] else 1
    suggestion = "" if flag == 0 else out.get("suggestion", "")

    rows.append({
        "old_value": name,
        "flag": flag,
        "suggestion": suggestion,
        "reason": out.get("reason", ""),
        "nearest_used": ", ".join(nearest)
    })


# -------------------------
# SAVE OUTPUT
# -------------------------
final = pd.DataFrame(rows)
final.to_excel(OUTPUT_FULL, index=False)
print(f"\nSaved FULL output → {OUTPUT_FULL}")

mistakes = final[final["flag"] == 1]
mistakes.to_excel(OUTPUT_MISTAKES, index=False)
print(f"Saved ONLY mistakes → {OUTPUT_MISTAKES}")

print("\nDONE ✔🚀")
