# =============================================================================
# ENTRAÎNEMENT DU MODÈLE A — Classification des Dialogue Acts
# Fichier : src/model_a/train_classifier.py
# =============================================================================
# Ce script :
#   1. Charge et prétraite le dataset SWDA (via clean_text.py)
#   2. Vectorise le texte avec TF-IDF
#   3. Entraîne un classifieur LinearSVC
#   4. Évalue le modèle sur un jeu de test
#   5. Sauvegarde le modèle entraîné pour une utilisation future

import os
import sys

import joblib
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# On ajoute le dossier racine du projet au chemin Python pour pouvoir
# importer notre module de prétraitement (src/preprocessing/clean_text.py)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.preprocessing.clean_text import preparer_dataset_swda

# =============================================================================
# ÉTAPE 1 : CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# =============================================================================

print("=" * 60)
print("ÉTAPE 1 : Chargement et prétraitement du dataset SWDA")
print("=" * 60)

# Chargement du dataset SWDA depuis HuggingFace
# Note : datasets==2.18.0 requis (trust_remote_code=True)
dataset = load_dataset("swda", trust_remote_code=True)

# Appel de notre fonction de prétraitement :
#   - Mapping des 43 labels → 11 macro-classes
#   - Suppression du BRUIT
#   - Nettoyage du texte avec spaCy (lemmatisation, stop words)
# Résultat : DataFrame avec colonnes 'texte_nettoye' et 'macro_classe'
df = preparer_dataset_swda(dataset)
df["contient_point_interrogation"] = df["text"].apply(
    lambda x: 1 if "?" in str(x) else 0
)


# =============================================================================
# ÉTAPE 2 : SÉPARATION ENTRAÎNEMENT / TEST
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 2 : Séparation entraînement / test (80% / 20%)")
print("=" * 60)

# X contient les textes (les features, ce qu'on donne au modèle)
# y contient les labels (ce que le modèle doit prédire)
X = df[["texte_nettoye", "contient_point_interrogation"]]
y = df["macro_classe"]

# train_test_split divise le dataset aléatoirement :
#   - test_size=0.2    : 20% des données pour le test
#   - random_state=42  : graine aléatoire fixe pour la reproductibilité
#                        (42 est une convention, ça donne toujours le même split)
#   - stratify=y       : conserve les mêmes proportions de classes dans chaque split
#                        (crucial avec nos classes déséquilibrées !)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Taille entraînement : {len(X_train)} répliques")
print(f"Taille test         : {len(X_test)} répliques")


# =============================================================================
# ÉTAPE 3 : CONSTRUCTION DU PIPELINE (TF-IDF + LinearSVC)
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 3 : Construction du pipeline TF-IDF + LinearSVC")
print("=" * 60)

# Un Pipeline scikit-learn enchaîne des étapes de transformation + modèle.
# Avantage : on peut entraîner et prédire en une seule commande.

preprocessor = ColumnTransformer(
    transformers=[
        # 1. On applique le TF-IDF uniquement sur la colonne 'texte_nettoye'
        (
            "tfidf",
            TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True),
            "texte_nettoye",
        ),
        # 2. On laisse passer la colonne 'contient_point_interrogation' telle quelle (sans la modifier)
        ("features_supp", "passthrough", ["contient_point_interrogation"]),
    ]
)

pipeline = Pipeline(
    [
        (
            "preprocessor",
            preprocessor,
        ),  # Le préprocesseur qui combine le TF-IDF et la feature "?"
        ("clf", LinearSVC(class_weight="balanced", max_iter=5000)),
    ]
)

print("Pipeline créé : TfidfVectorizer(ngrams 1-2, 50k features) → LinearSVC(balanced)")


# =============================================================================
# ÉTAPE 4 : ENTRAÎNEMENT
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 4 : Entraînement du modèle...")
print("=" * 60)

# pipeline.fit() enclenche les deux étapes en séquence :
#   1. TfidfVectorizer apprend le vocabulaire sur X_train et transforme les textes
#   2. LinearSVC s'entraîne sur les vecteurs résultants
pipeline.fit(X_train, y_train)

print("Entraînement terminé !")


# =============================================================================
# ÉTAPE 5 : ÉVALUATION SUR LE JEU DE TEST
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 5 : Évaluation sur le jeu de test")
print("=" * 60)

# pipeline.predict() applique le pipeline sur les données de test :
#   1. TfidfVectorizer transforme X_test avec le vocabulaire appris
#   2. LinearSVC prédit la classe pour chaque vecteur
y_pred = pipeline.predict(X_test)

# classification_report affiche pour chaque classe :
#   - precision : quand le modèle dit "ORDRE", il a raison x% du temps
#   - recall    : parmi tous les vrais ORDRE, le modèle en trouve x%
#   - f1-score  : moyenne harmonique de precision et recall
#   - support   : nombre d'exemples de cette classe dans le jeu de test
print(classification_report(y_test, y_pred))


# =============================================================================
# ÉTAPE 6 : SAUVEGARDE DU MODÈLE
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 6 : Sauvegarde du modèle")
print("=" * 60)

# joblib.dump() sérialise le pipeline complet (TF-IDF + LinearSVC) dans un fichier.
# On pourra le recharger plus tard avec joblib.load() pour faire des prédictions
# sur le Cornell Movie-Dialogs Corpus sans avoir à ré-entraîner.
chemin_modele = os.path.join(os.path.dirname(__file__), "modele_dialogue_acts.joblib")
joblib.dump(pipeline, chemin_modele)

print(f"Modèle sauvegardé dans : {chemin_modele}")
print("\nPour le recharger plus tard :")
print("  import joblib")
print(f"  pipeline = joblib.load('{chemin_modele}')")
print("  prediction = pipeline.predict(['your cleaned text here'])")
