# =============================================================================
# MODÈLE B — EXTRACTION DE THÈMES PAR LDA
# Fichier : src/model_b/extract_topics.py
# =============================================================================
# Ce fichier utilise LDA (Latent Dirichlet Allocation) pour découvrir
# automatiquement des thèmes latents dans les répliques de films.
#
# LDA est un algorithme NON-SUPERVISÉ : il ne reçoit aucun label.
# Il trouve des groupes de mots qui apparaissent souvent ensemble
# et les interprète comme des "thèmes" (ex: guerre, famille, amour...).
#
# Chaque réplique reçoit une distribution de probabilités sur les thèmes.
# Ex: "I love you so much" → 80% Amour, 10% Famille, 10% Autre

import os
import sys
import joblib
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Ajout du dossier racine au path pour pouvoir importer clean_text
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.clean_text import nettoyer_texte


# =============================================================================
# CONSTANTES
# =============================================================================

# Nombre de thèmes à découvrir — à ajuster selon les résultats
N_THEMES = 12

# Nombre de mots les plus représentatifs à afficher par thème
N_MOTS_PAR_THEME = 15

# Chemin de sauvegarde du modèle LDA et du vectoriseur
CHEMIN_MODELE_LDA = os.path.join(os.path.dirname(__file__), "modele_lda.joblib")
CHEMIN_VECTORISEUR = os.path.join(os.path.dirname(__file__), "vectoriseur_lda.joblib")


# =============================================================================
# FONCTION 1 : nettoyage des répliques pour LDA
# =============================================================================

def nettoyer_repliques(df, col_texte="text"):
    """
    Nettoie les répliques d'un DataFrame avec spaCy pour préparer le LDA.

    On réutilise la fonction nettoyer_texte() de clean_text.py pour avoir
    un preprocessing identique à celui du Modèle A (cohérence du pipeline).

    Paramètres :
      df (DataFrame)   : le DataFrame contenant les répliques
      col_texte (str)  : nom de la colonne contenant le texte brut

    Retourne :
      Series Pandas : les répliques nettoyées (lemmes séparés par des espaces)
    """
    print("Chargement du modèle spaCy...")
    # disable=["parser", "ner"] : on n'a besoin que du tagger pour la lemmatisation
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    print(f"Nettoyage de {len(df)} répliques avec spaCy...")
    # Application de nettoyer_texte() sur chaque réplique
    textes_nettoyes = df[col_texte].apply(
        lambda t: nettoyer_texte(str(t), nlp)
    )

    # Suppression des répliques vides après nettoyage
    nb_avant = len(textes_nettoyes)
    textes_nettoyes = textes_nettoyes[textes_nettoyes.str.strip() != ""]
    nb_apres = len(textes_nettoyes)
    print(f"Répliques supprimées (vides après nettoyage) : {nb_avant - nb_apres}")

    return textes_nettoyes


# =============================================================================
# FONCTION 2 : entraînement du modèle LDA
# =============================================================================

def entrainer_lda(textes_nettoyes, n_themes=N_THEMES):
    """
    Entraîne un modèle LDA sur les répliques nettoyées.

    Pipeline :
      1. CountVectorizer : convertit les textes en matrice de comptes de mots
         (LDA utilise des comptes bruts, PAS du TF-IDF)
      2. LatentDirichletAllocation : découvre les thèmes latents

    Paramètres :
      textes_nettoyes (Series) : répliques nettoyées (sortie de nettoyer_repliques)
      n_themes (int)           : nombre de thèmes à découvrir

    Retourne :
      tuple : (modele_lda, vectoriseur, matrice_comptes)
    """
    # --- Étape 1 : vectorisation par comptage ---
    # max_df=0.95 : ignore les mots présents dans +95% des documents (trop communs)
    # min_df=10   : ignore les mots présents dans moins de 10 documents (trop rares)
    # max_features=5000 : limite le vocabulaire aux 5000 mots les plus fréquents
    print("\nVectorisation (CountVectorizer)...")
    vectoriseur = CountVectorizer(
        max_df=0.95,
        min_df=10,
        max_features=5000
    )
    matrice_comptes = vectoriseur.fit_transform(textes_nettoyes)
    print(f"Matrice de comptes : {matrice_comptes.shape[0]} documents × "
          f"{matrice_comptes.shape[1]} mots")

    # --- Étape 2 : entraînement LDA ---
    # n_components : nombre de thèmes à découvrir
    # random_state=42 : pour la reproductibilité
    # max_iter=20 : nombre d'itérations (augmenter si les thèmes sont flous)
    # n_jobs=-1 : utilise tous les cœurs du processeur
    print(f"\nEntraînement LDA avec {n_themes} thèmes...")
    modele_lda = LatentDirichletAllocation(
        n_components=n_themes,
        random_state=42,
        max_iter=20,
        n_jobs=-1,
        verbose=1
    )
    modele_lda.fit(matrice_comptes)
    print("Entraînement terminé.")

    return modele_lda, vectoriseur, matrice_comptes


# =============================================================================
# FONCTION 3 : affichage des thèmes découverts
# =============================================================================

def afficher_themes(modele_lda, vectoriseur, n_mots=N_MOTS_PAR_THEME):
    """
    Affiche les mots les plus représentatifs de chaque thème.

    C'est l'étape d'INTERPRÉTATION : on regarde les mots de chaque thème
    et on leur donne un nom humain (ex: "Guerre", "Amour", "Travail").

    Paramètres :
      modele_lda   : le modèle LDA entraîné
      vectoriseur  : le CountVectorizer utilisé pour l'entraînement
      n_mots (int) : nombre de mots à afficher par thème
    """
    # Récupération de la liste des mots du vocabulaire
    noms_mots = vectoriseur.get_feature_names_out()

    print("\n" + "=" * 70)
    print(f"THÈMES DÉCOUVERTS PAR LDA ({modele_lda.n_components} thèmes)")
    print("=" * 70)

    for i, theme in enumerate(modele_lda.components_):
        # theme est un vecteur de poids : plus le poids est élevé,
        # plus le mot est représentatif du thème
        # argsort() trie les indices par poids croissant
        # [:-n_mots-1:-1] prend les n_mots derniers (les plus importants)
        indices_top = theme.argsort()[:-n_mots - 1:-1]
        mots_top = [noms_mots[j] for j in indices_top]
        print(f"\nThème {i:>2} : {', '.join(mots_top)}")


# =============================================================================
# FONCTION 4 : assignation du thème dominant à chaque réplique
# =============================================================================

def assigner_themes(modele_lda, vectoriseur, textes_nettoyes):
    """
    Assigne le thème dominant à chaque réplique.

    Pour chaque réplique, LDA calcule une distribution de probabilités
    sur tous les thèmes. On prend le thème avec la probabilité la plus élevée.

    Paramètres :
      modele_lda       : le modèle LDA entraîné
      vectoriseur      : le CountVectorizer utilisé
      textes_nettoyes  : les répliques nettoyées

    Retourne :
      Series Pandas : le numéro du thème dominant pour chaque réplique
    """
    # Vectorisation des textes avec le même vocabulaire
    matrice = vectoriseur.transform(textes_nettoyes)

    # transform() retourne une matrice (n_docs × n_themes) de probabilités
    distributions = modele_lda.transform(matrice)

    # argmax(axis=1) : pour chaque ligne, l'indice du thème avec la proba max
    themes_dominants = distributions.argmax(axis=1)

    return pd.Series(themes_dominants, index=textes_nettoyes.index, name="theme_dominant")


# =============================================================================
# FONCTION 5 : sauvegarde du modèle LDA et du vectoriseur
# =============================================================================

def sauvegarder_modele(modele_lda, vectoriseur):
    """
    Sauvegarde le modèle LDA et le vectoriseur pour réutilisation ultérieure.

    On sauvegarde les deux objets séparément car ils sont nécessaires
    ensemble pour faire des prédictions sur de nouveaux textes.

    Paramètres :
      modele_lda  : le modèle LDA entraîné
      vectoriseur : le CountVectorizer utilisé
    """
    joblib.dump(modele_lda, CHEMIN_MODELE_LDA)
    print(f"Modèle LDA sauvegardé : {CHEMIN_MODELE_LDA}")

    joblib.dump(vectoriseur, CHEMIN_VECTORISEUR)
    print(f"Vectoriseur sauvegardé : {CHEMIN_VECTORISEUR}")


# =============================================================================
# FONCTION 6 : chargement du modèle LDA sauvegardé
# =============================================================================

def charger_modele_lda():
    """
    Charge le modèle LDA et le vectoriseur depuis les fichiers .joblib.

    Retourne :
      tuple : (modele_lda, vectoriseur)
    """
    modele_lda = joblib.load(CHEMIN_MODELE_LDA)
    vectoriseur = joblib.load(CHEMIN_VECTORISEUR)
    print("Modèle LDA et vectoriseur chargés.")
    return modele_lda, vectoriseur


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    from src.preprocessing.load_cornell import charger_cornell

    print("=== MODÈLE B — EXTRACTION DE THÈMES PAR LDA ===\n")

    # --- Étape 1 : chargement du corpus Cornell ---
    df_cornell = charger_cornell()

    # On garde uniquement les répliques avec genre connu (m ou f)
    # car ce sont celles qu'on analysera dans l'étape croisée
    df_analyse = df_cornell[df_cornell["character_gender"].isin(["m", "f"])].copy()
    print(f"\nRépliques avec genre connu : {len(df_analyse)}")

    # --- Étape 2 : nettoyage des répliques ---
    textes_propres = nettoyer_repliques(df_analyse, col_texte="text")

    # --- Étape 3 : entraînement LDA ---
    modele_lda, vectoriseur, matrice = entrainer_lda(textes_propres, n_themes=N_THEMES)

    # --- Étape 4 : affichage des thèmes ---
    afficher_themes(modele_lda, vectoriseur)

    # --- Étape 5 : assignation des thèmes dominants ---
    themes = assigner_themes(modele_lda, vectoriseur, textes_propres)
    df_analyse.loc[textes_propres.index, "theme_dominant"] = themes

    print("\n--- Distribution des thèmes ---")
    print(df_analyse["theme_dominant"].value_counts().sort_index())

    # --- Étape 6 : sauvegarde du modèle ---
    sauvegarder_modele(modele_lda, vectoriseur)

    print("\n=== Modèle B terminé ===")
