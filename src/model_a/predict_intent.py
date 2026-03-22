# =============================================================================
# PRÉDICTION D'INTENTIONS — Modèle A
# Fichier : src/model_a/predict_intent.py
# =============================================================================
# Ce fichier contient les fonctions pour utiliser le modèle entraîné afin
# de prédire la macro-classe (ORDRE, QUESTION, STATEMENT...) d'une réplique.
#
# RÈGLE FONDAMENTALE : le preprocessing ici DOIT être identique à celui
# appliqué pendant l'entraînement (même nettoyage spaCy).

import os
import joblib
import spacy
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.clean_text import nettoyer_texte


# =============================================================================
# FONCTION 1 : chargement du modèle sauvegardé
# =============================================================================

def charger_modele(chemin=None):
    """
    Charge le pipeline entraîné (TF-IDF + LinearSVC) depuis le fichier .joblib.

    Important : appeler cette fonction UNE SEULE FOIS au début du programme.
    Le chargement prend quelques secondes — éviter de le répéter à chaque prédiction.

    Paramètre :
      chemin (str) : chemin vers le fichier .joblib
                     Par défaut : src/model_a/modele_dialogue_acts.joblib

    Retourne :
      pipeline scikit-learn : le modèle prêt à l'emploi
    """
    if chemin is None:
        # Chemin par défaut : même dossier que ce fichier
        chemin = os.path.join(os.path.dirname(__file__), "modele_dialogue_acts.joblib")

    print(f"Chargement du modèle depuis : {chemin}")
    pipeline = joblib.load(chemin)
    print("Modèle chargé avec succès.")
    return pipeline


# =============================================================================
# FONCTION 2 : prédiction sur une seule réplique
# =============================================================================

def predire_intention(texte_brut, pipeline, nlp):
    """
    Prédit la macro-classe d'une réplique brute.

    Étapes :
      1. Nettoyage du texte brut avec spaCy (même preprocessing qu'à l'entraînement)
      2. Prédiction via le pipeline (TF-IDF → LinearSVC)

    Paramètres :
      texte_brut (str) : la réplique brute (ex: "Get in the car right now!")
      pipeline         : le pipeline scikit-learn chargé avec charger_modele()
      nlp              : le modèle spaCy chargé

    Retourne :
      str : la macro-classe prédite (ex: "ORDRE", "QUESTION", "STATEMENT"...)
            ou "VIDE" si le texte est vide après nettoyage
    """

    # Étape 1 : nettoyage identique à l'entraînement
    texte_nettoye = nettoyer_texte(str(texte_brut), nlp)

    # Si le texte est vide après nettoyage (ex: réplique = "Uh..."),
    # on retourne une valeur spéciale plutôt que de provoquer une erreur
    if not texte_nettoye.strip():
        return "VIDE"

    # Étape 2 : prédiction
    # Le pipeline utilise un ColumnTransformer — il attend un DataFrame
    # avec les mêmes colonnes que lors de l'entraînement
    contient_point = 1 if "?" in str(texte_brut) else 0
    df_input = pd.DataFrame({
        "texte_nettoye": [texte_nettoye],
        "contient_point_interrogation": [contient_point]
    })
    prediction = pipeline.predict(df_input)[0]

    return prediction


# =============================================================================
# FONCTION 3 : prédiction sur tout un DataFrame
# =============================================================================

def predire_sur_dataframe(df, col_texte, pipeline, nlp):
    """
    Applique la prédiction d'intention sur toute une colonne d'un DataFrame.

    C'est la fonction principale pour l'analyse du Cornell Movie-Dialogs Corpus :
    on lui passe le DataFrame des répliques, et elle ajoute une colonne
    'intention_predite' avec la macro-classe de chaque réplique.

    Paramètres :
      df (DataFrame)   : le DataFrame contenant les répliques
      col_texte (str)  : le nom de la colonne contenant le texte brut
      pipeline         : le pipeline chargé avec charger_modele()
      nlp              : le modèle spaCy chargé

    Retourne :
      DataFrame : le même DataFrame avec une colonne 'intention_predite' ajoutée
    """

    print(f"Prédiction sur {len(df)} répliques...")

    # On applique predire_intention() sur chaque ligne du DataFrame
    # lambda x : fonction anonyme qui appelle predire_intention pour chaque texte
    df = df.copy()  # on ne modifie pas le DataFrame original
    df["intention_predite"] = df[col_texte].apply(
        lambda texte: predire_intention(texte, pipeline, nlp)
    )

    print("Prédiction terminée.")
    print("\nDistribution des intentions prédites :")
    print(df["intention_predite"].value_counts())

    return df


# =============================================================================
# TEST MANUEL (exécuté seulement si on lance ce fichier directement)
# =============================================================================

if __name__ == "__main__":

    # --- Chargement des outils ---
    pipeline = charger_modele()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # --- Répliques de test représentatives ---
    # On teste sur des répliques typiques pour valider que le modèle a du sens.
    # Ces exemples sont proches de ce qu'on trouvera dans le Cornell corpus.
    exemples = [
        ("Get in the car right now!",           "ORDRE attendu"),
        ("Do you want some coffee?",             "QUESTION attendu"),
        ("I think this is a great idea.",        "OPINION attendu"),
        ("I'm so sorry for what happened.",      "POLITESSE attendu"),
        ("Yeah, absolutely.",                    "ACCORD attendu"),
        ("Uh-huh.",                              "BACKCHANNEL attendu"),
        ("That's not right at all.",             "DESACCORD attendu"),
        ("Life is complicated, you know.",       "STATEMENT attendu"),
        ("You never listen to me!",              "PLAINTE attendu"),
    ]

    print("\n" + "=" * 65)
    print(f"{'RÉPLIQUE':<40} {'ATTENDU':<20} {'PRÉDIT'}")
    print("=" * 65)

    for texte, attendu in exemples:
        predit = predire_intention(texte, pipeline, nlp)
        # Indicateur visuel : ✓ si correct, ✗ si incorrect
        attendu_court = attendu.replace(" attendu", "")
        icone = "✓" if predit == attendu_court else "✗"
        print(f"{texte:<40} {attendu:<20} {predit} {icone}")
