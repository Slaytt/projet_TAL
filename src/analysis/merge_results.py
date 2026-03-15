# =============================================================================
# ANALYSE CROISÉE — INTENTIONS × THÈMES × GENRE
# Fichier : src/analysis/merge_results.py
# =============================================================================
# Ce fichier est le CŒUR du projet : il croise les résultats du Modèle A
# (intentions de dialogue) et du Modèle B (thèmes LDA) avec le genre des
# personnages pour mettre en évidence des biais de genre dans le cinéma.
#
# Pipeline :
#   1. Charger le corpus Cornell
#   2. Appliquer le Modèle A (prédiction d'intentions)
#   3. Appliquer le Modèle B (assignation de thèmes)
#   4. Croiser avec le genre du personnage
#   5. Calculer des statistiques et produire des visualisations

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-interactif pour sauvegarder les figures
import matplotlib.pyplot as plt

# Ajout du dossier racine au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.load_cornell import charger_cornell
from src.preprocessing.clean_text import nettoyer_texte
from src.model_a.predict_intent import charger_modele, predire_sur_dataframe
from src.model_b.extract_topics import (
    charger_modele_lda, nettoyer_repliques, assigner_themes
)
import spacy

# Dossier de sortie pour les graphiques
DOSSIER_RESULTATS = os.path.join(os.path.dirname(__file__), "..", "..", "resultats")


# =============================================================================
# FONCTION 1 : construction du DataFrame complet (intentions + thèmes + genre)
# =============================================================================

def construire_dataframe_complet():
    """
    Construit le DataFrame final en appliquant les deux modèles sur le Cornell.

    Étapes :
      1. Chargement du corpus Cornell (répliques avec genre connu)
      2. Prédiction des intentions avec le Modèle A (LinearSVC)
      3. Nettoyage + assignation des thèmes avec le Modèle B (LDA)
      4. Fusion en un seul DataFrame

    Retourne :
      DataFrame avec les colonnes :
        - text : réplique brute
        - character_gender : 'm' ou 'f'
        - movie_title, movie_year
        - intention_predite : macro-classe du Modèle A
        - texte_nettoye : réplique nettoyée par spaCy
        - theme_dominant : numéro du thème LDA
    """
    # --- Étape 1 : chargement du corpus Cornell ---
    print("=" * 60)
    print("ÉTAPE 1 : Chargement du corpus Cornell")
    print("=" * 60)
    df = charger_cornell()

    # On garde uniquement les répliques avec genre connu
    df = df[df["character_gender"].isin(["m", "f"])].copy()
    df = df.reset_index(drop=True)
    print(f"\nRépliques avec genre connu : {len(df)}")

    # --- Étape 2 : prédiction des intentions (Modèle A) ---
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Prédiction des intentions (Modèle A)")
    print("=" * 60)
    pipeline_a = charger_modele()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    df = predire_sur_dataframe(df, "text", pipeline_a, nlp)

    # Suppression des répliques classées "VIDE" (texte vide après nettoyage)
    nb_vides = (df["intention_predite"] == "VIDE").sum()
    df = df[df["intention_predite"] != "VIDE"].copy()
    print(f"Répliques VIDE supprimées : {nb_vides}")

    # --- Étape 3 : assignation des thèmes (Modèle B) ---
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Assignation des thèmes (Modèle B — LDA)")
    print("=" * 60)
    modele_lda, vectoriseur = charger_modele_lda()

    # Nettoyage des répliques pour le LDA
    textes_propres = nettoyer_repliques(df, col_texte="text")

    # On ne garde que les répliques non-vides après nettoyage
    df = df.loc[textes_propres.index].copy()
    df["texte_nettoye"] = textes_propres

    # Assignation du thème dominant
    themes = assigner_themes(modele_lda, vectoriseur, textes_propres)
    df["theme_dominant"] = themes.values

    print(f"\nDataFrame final : {len(df)} répliques")
    return df


# =============================================================================
# FONCTION 2 : analyse des intentions par genre
# =============================================================================

def analyser_intentions_par_genre(df):
    """
    Calcule la distribution des intentions par genre et affiche les résultats.

    Pour chaque intention (ORDRE, QUESTION, POLITESSE...), on calcule
    le pourcentage prononcé par des hommes vs des femmes.

    Paramètre :
      df (DataFrame) : le DataFrame complet (sortie de construire_dataframe_complet)

    Retourne :
      DataFrame : tableau croisé intentions × genre (en pourcentages)
    """
    print("\n" + "=" * 60)
    print("ANALYSE 1 : Distribution des intentions par genre")
    print("=" * 60)

    # Tableau croisé : pour chaque intention, nombre de répliques par genre
    tableau = pd.crosstab(
        df["intention_predite"],
        df["character_gender"],
        normalize="index"  # normalisation par ligne → pourcentages
    ) * 100

    # Renommage des colonnes pour la lisibilité
    tableau = tableau.rename(columns={"m": "Hommes (%)", "f": "Femmes (%)"})

    # Ajout du nombre total de répliques par intention
    tableau["Total"] = df["intention_predite"].value_counts()

    # Tri par écart le plus marqué (différence hommes-femmes)
    tableau["Écart"] = abs(tableau["Hommes (%)"] - tableau["Femmes (%)"])
    tableau = tableau.sort_values("Écart", ascending=False)

    print("\n" + tableau.round(1).to_string())

    return tableau


# =============================================================================
# FONCTION 3 : analyse des thèmes par genre
# =============================================================================

def analyser_themes_par_genre(df):
    """
    Calcule la distribution des thèmes LDA par genre.

    Pour chaque thème (0 à 11), on calcule le pourcentage de répliques
    prononcées par des hommes vs des femmes.

    Paramètre :
      df (DataFrame) : le DataFrame complet

    Retourne :
      DataFrame : tableau croisé thèmes × genre (en pourcentages)
    """
    print("\n" + "=" * 60)
    print("ANALYSE 2 : Distribution des thèmes par genre")
    print("=" * 60)

    # Tableau croisé thèmes × genre
    tableau = pd.crosstab(
        df["theme_dominant"],
        df["character_gender"],
        normalize="index"
    ) * 100

    tableau = tableau.rename(columns={"m": "Hommes (%)", "f": "Femmes (%)"})
    tableau["Total"] = df["theme_dominant"].value_counts()
    tableau["Écart"] = abs(tableau["Hommes (%)"] - tableau["Femmes (%)"])
    tableau = tableau.sort_values("Écart", ascending=False)

    print("\n" + tableau.round(1).to_string())

    return tableau


# =============================================================================
# FONCTION 4 : analyse temporelle (par décennie)
# =============================================================================

def analyser_evolution_temporelle(df):
    """
    Analyse l'évolution de la répartition des intentions par genre au fil du temps.

    On regroupe les films par décennie et on observe si les biais
    évoluent (ex: les femmes reçoivent-elles plus d'ORDRES dans les films récents ?).

    Paramètre :
      df (DataFrame) : le DataFrame complet

    Retourne :
      DataFrame : intentions × genre × décennie
    """
    print("\n" + "=" * 60)
    print("ANALYSE 3 : Évolution temporelle (par décennie)")
    print("=" * 60)

    # Création de la colonne décennie
    df_temp = df.dropna(subset=["movie_year"]).copy()
    df_temp["decennie"] = (df_temp["movie_year"] // 10 * 10).astype(int)

    # On ne garde que les décennies avec assez de données
    counts_dec = df_temp["decennie"].value_counts()
    decennies_valides = counts_dec[counts_dec > 1000].index
    df_temp = df_temp[df_temp["decennie"].isin(decennies_valides)]

    # Pour chaque décennie, proportion de femmes par intention
    resultats = []
    for dec in sorted(df_temp["decennie"].unique()):
        sous_df = df_temp[df_temp["decennie"] == dec]
        for intention in sorted(sous_df["intention_predite"].unique()):
            sous_intention = sous_df[sous_df["intention_predite"] == intention]
            pct_femmes = (sous_intention["character_gender"] == "f").mean() * 100
            resultats.append({
                "decennie": dec,
                "intention": intention,
                "pct_femmes": round(pct_femmes, 1),
                "nb_repliques": len(sous_intention)
            })

    df_evol = pd.DataFrame(resultats)

    # Affichage pivot : décennies en lignes, intentions en colonnes
    pivot = df_evol.pivot(index="decennie", columns="intention", values="pct_femmes")
    print("\n% de répliques prononcées par des FEMMES, par décennie :\n")
    print(pivot.round(1).to_string())

    return df_evol


# =============================================================================
# FONCTION 5 : génération des graphiques
# =============================================================================

def generer_graphiques(df, tab_intentions, tab_themes):
    """
    Génère et sauvegarde les graphiques de l'analyse croisée.

    Graphiques produits :
      1. Barplot : intentions par genre
      2. Barplot : thèmes par genre
      3. Proportion globale hommes/femmes

    Paramètres :
      df              : le DataFrame complet
      tab_intentions  : tableau croisé intentions × genre
      tab_themes      : tableau croisé thèmes × genre
    """
    # Création du dossier de résultats
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)

    # --- Graphique 1 : intentions par genre ---
    fig, ax = plt.subplots(figsize=(12, 6))
    tab_plot = tab_intentions[["Hommes (%)", "Femmes (%)"]].sort_index()
    tab_plot.plot(kind="barh", ax=ax, color=["#4A90D9", "#E8737A"])
    ax.set_xlabel("Pourcentage (%)")
    ax.set_ylabel("Intention")
    ax.set_title("Distribution des intentions par genre — Cornell Movie-Dialogs")
    ax.legend(loc="lower right")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)  # ligne de parité
    plt.tight_layout()
    chemin1 = os.path.join(DOSSIER_RESULTATS, "intentions_par_genre.png")
    plt.savefig(chemin1, dpi=150)
    plt.close()
    print(f"\nGraphique sauvegardé : {chemin1}")

    # --- Graphique 2 : thèmes par genre ---
    fig, ax = plt.subplots(figsize=(12, 6))
    tab_plot2 = tab_themes[["Hommes (%)", "Femmes (%)"]].sort_index()
    tab_plot2.index = [f"Thème {int(i)}" for i in tab_plot2.index]
    tab_plot2.plot(kind="barh", ax=ax, color=["#4A90D9", "#E8737A"])
    ax.set_xlabel("Pourcentage (%)")
    ax.set_ylabel("Thème LDA")
    ax.set_title("Distribution des thèmes par genre — Cornell Movie-Dialogs")
    ax.legend(loc="lower right")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    chemin2 = os.path.join(DOSSIER_RESULTATS, "themes_par_genre.png")
    plt.savefig(chemin2, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {chemin2}")

    # --- Graphique 3 : proportion globale ---
    fig, ax = plt.subplots(figsize=(6, 6))
    dist = df["character_gender"].value_counts()
    labels = ["Hommes", "Femmes"]
    couleurs = ["#4A90D9", "#E8737A"]
    ax.pie(dist.values, labels=labels, colors=couleurs, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 14})
    ax.set_title("Répartition globale des répliques par genre")
    plt.tight_layout()
    chemin3 = os.path.join(DOSSIER_RESULTATS, "repartition_globale_genre.png")
    plt.savefig(chemin3, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {chemin3}")


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ANALYSE CROISÉE — INTENTIONS × THÈMES × GENRE")
    print("=" * 60)

    # --- Construction du DataFrame complet ---
    df_complet = construire_dataframe_complet()

    # --- Analyses statistiques ---
    tab_intentions = analyser_intentions_par_genre(df_complet)
    tab_themes = analyser_themes_par_genre(df_complet)
    df_evol = analyser_evolution_temporelle(df_complet)

    # --- Génération des graphiques ---
    generer_graphiques(df_complet, tab_intentions, tab_themes)

    # --- Sauvegarde du DataFrame complet pour le notebook final ---
    chemin_csv = os.path.join(DOSSIER_RESULTATS, "resultats_complets.csv")
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    df_complet.to_csv(chemin_csv, index=False)
    print(f"\nDataFrame complet sauvegardé : {chemin_csv}")

    print("\n" + "=" * 60)
    print("ANALYSE CROISÉE TERMINÉE")
    print("=" * 60)
