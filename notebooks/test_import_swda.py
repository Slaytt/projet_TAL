# =============================================================================
# TEST D'IMPORT DU DATASET SWDA (Switchboard Dialogue Act Corpus)
# =============================================================================
# Ce script vérifie que le dataset se charge bien et nous permet de
# comprendre sa structure avant de commencer à modéliser.

from datasets import load_dataset
import pandas as pd

# -----------------------------------------------------------------------------
# 1. CHARGEMENT DU DATASET
# -----------------------------------------------------------------------------
# Le dataset "swda" est hébergé sur HuggingFace.
# trust_remote_code=True est nécessaire car le dataset utilise un script
# de chargement personnalisé.
print("Chargement du dataset SWDA en cours (peut prendre 1-2 min)...")
dataset = load_dataset("swda", trust_remote_code=True)

print("\n=== STRUCTURE DU DATASET ===")
# Le dataset est divisé en 3 parties : train, validation, test
print(dataset)

# -----------------------------------------------------------------------------
# 2. CONVERSION EN DATAFRAME PANDAS
# -----------------------------------------------------------------------------
# On travaille avec la partie "train" pour l'exploration.
df = pd.DataFrame(dataset["train"])

print("\n=== COLONNES DISPONIBLES ===")
print(df.columns.tolist())

print("\n=== 5 PREMIÈRES LIGNES ===")
# Les deux colonnes clés :
# - "text"         : la réplique brute
# - "damsl_act_tag": l'intention (ex: "sd" pour statement, "qy" pour yes/no question)
print(df[["text", "damsl_act_tag"]].head())

# -----------------------------------------------------------------------------
# 3. DISTRIBUTION DES LABELS (intentions)
# -----------------------------------------------------------------------------
# Si certaines intentions sont très rares, notre modèle aura du mal à les apprendre.
print(f"\n=== DISTRIBUTION DES INTENTIONS (damsl_act_tag) ===")
print(f"Nombre total de classes : {df['damsl_act_tag'].nunique()}")
print("\nTop 20 des intentions les plus fréquentes :")
print(df["damsl_act_tag"].value_counts().head(20))
