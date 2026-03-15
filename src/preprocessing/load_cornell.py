# =============================================================================
# CHARGEMENT DU CORNELL MOVIE-DIALOGS CORPUS
# Fichier : src/preprocessing/load_cornell.py
# =============================================================================
# Ce fichier télécharge, parse et structure le Cornell Movie-Dialogs Corpus
# en un DataFrame Pandas prêt à être analysé par les Modèles A et B.
#
# Le corpus contient ~300 000 répliques issues de ~600 films, avec des
# métadonnées sur les personnages (genre) et les films (titre, année, genre).
#
# Source : https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

import os
import io
import zipfile
import urllib.request
import pandas as pd

# =============================================================================
# CONSTANTES
# =============================================================================

# URL officielle du corpus Cornell
URL_CORNELL = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"

# Dossier où stocker les fichiers bruts téléchargés
DOSSIER_RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# Séparateur utilisé dans les fichiers du corpus Cornell
# Chaque colonne est séparée par " +++$+++ "
SEPARATEUR = " +++$+++ "


# =============================================================================
# FONCTION 1 : téléchargement et extraction du corpus
# =============================================================================

def telecharger_cornell(dossier_destination=None):
    """
    Télécharge le corpus Cornell depuis le site officiel et l'extrait.

    Le corpus est un fichier .zip d'environ 10 Mo. Il contient un dossier
    'cornell movie-dialogs corpus' avec plusieurs fichiers texte.

    Si le corpus est déjà présent localement, on saute le téléchargement.

    Paramètre :
      dossier_destination (str) : dossier où extraire les fichiers
                                  Par défaut : data/raw/

    Retourne :
      str : chemin vers le dossier extrait contenant les fichiers .txt
    """
    if dossier_destination is None:
        dossier_destination = DOSSIER_RAW

    # Chemin attendu du dossier extrait
    dossier_corpus = os.path.join(dossier_destination, "cornell movie-dialogs corpus")

    # Vérification : le corpus est-il déjà téléchargé ?
    fichier_test = os.path.join(dossier_corpus, "movie_lines.txt")
    if os.path.exists(fichier_test):
        print(f"Corpus Cornell déjà présent dans : {dossier_corpus}")
        return dossier_corpus

    # Création du dossier de destination si nécessaire
    os.makedirs(dossier_destination, exist_ok=True)

    # Téléchargement du fichier zip
    print(f"Téléchargement du corpus Cornell (~10 Mo)...")
    print(f"URL : {URL_CORNELL}")
    donnees_zip, _ = urllib.request.urlretrieve(URL_CORNELL)

    # Extraction du zip dans le dossier de destination
    print(f"Extraction dans : {dossier_destination}")
    with zipfile.ZipFile(donnees_zip, 'r') as z:
        z.extractall(dossier_destination)

    print("Téléchargement et extraction terminés.")
    return dossier_corpus


# =============================================================================
# FONCTION 2 : parsing d'un fichier Cornell (format +++$+++)
# =============================================================================

def parser_fichier_cornell(chemin_fichier, noms_colonnes):
    """
    Parse un fichier du corpus Cornell et retourne un DataFrame.

    Les fichiers du corpus utilisent le séparateur ' +++$+++ ' entre colonnes.
    L'encodage est iso-8859-1 (latin-1), pas UTF-8.

    Paramètres :
      chemin_fichier (str)  : chemin vers le fichier .txt
      noms_colonnes (list)  : liste des noms de colonnes attendues

    Retourne :
      DataFrame Pandas avec les colonnes spécifiées
    """
    lignes_parsees = []

    # Ouverture en latin-1 car le corpus contient des caractères spéciaux
    with open(chemin_fichier, 'r', encoding='iso-8859-1') as f:
        for ligne in f:
            # Découpage de la ligne selon le séparateur Cornell
            champs = ligne.strip().split(SEPARATEUR)

            # On ne garde que les lignes qui ont le bon nombre de colonnes
            if len(champs) == len(noms_colonnes):
                lignes_parsees.append(champs)

    # Construction du DataFrame
    df = pd.DataFrame(lignes_parsees, columns=noms_colonnes)
    print(f"  → {chemin_fichier.split('/')[-1]} : {len(df)} lignes chargées")

    return df


# =============================================================================
# FONCTION 3 : chargement complet du corpus Cornell
# =============================================================================

def charger_cornell(dossier_corpus=None):
    """
    Charge et fusionne les fichiers du corpus Cornell en un seul DataFrame.

    Fichiers parsés :
      - movie_lines.txt : les répliques (texte brut)
      - movie_characters_metadata.txt : métadonnées des personnages (genre M/F)
      - movie_titles_metadata.txt : métadonnées des films (titre, année)

    On fusionne ces trois fichiers via les identifiants de personnage et de film
    pour obtenir un DataFrame unique avec : texte, genre du personnage, titre, année.

    Paramètre :
      dossier_corpus (str) : chemin vers le dossier extrait du corpus
                             Si None, on télécharge d'abord le corpus

    Retourne :
      DataFrame Pandas avec les colonnes :
        - 'text'             : la réplique brute
        - 'character_gender' : genre du personnage ('m', 'f', ou '?')
        - 'movie_title'      : titre du film
        - 'movie_year'       : année de sortie du film (int ou NaN)
    """
    # --- Étape 0 : téléchargement si nécessaire ---
    if dossier_corpus is None:
        dossier_corpus = telecharger_cornell()

    print("\n=== Parsing des fichiers Cornell ===\n")

    # --- Étape 1 : parsing des répliques (movie_lines.txt) ---
    # Colonnes : lineID, characterID, movieID, characterName, text
    df_lines = parser_fichier_cornell(
        os.path.join(dossier_corpus, "movie_lines.txt"),
        ["line_id", "character_id", "movie_id", "character_name", "text"]
    )

    # --- Étape 2 : parsing des métadonnées des personnages ---
    # Colonnes : characterID, characterName, movieID, movieTitle, gender, creditPos
    df_characters = parser_fichier_cornell(
        os.path.join(dossier_corpus, "movie_characters_metadata.txt"),
        ["character_id", "character_name", "movie_id", "movie_title", "gender", "credit_position"]
    )

    # --- Étape 3 : parsing des métadonnées des films ---
    # Colonnes : movieID, movieTitle, movieYear, imdbRating, numImdbVotes, genres
    df_movies = parser_fichier_cornell(
        os.path.join(dossier_corpus, "movie_titles_metadata.txt"),
        ["movie_id", "movie_title", "movie_year", "imdb_rating", "num_imdb_votes", "genres"]
    )

    # --- Étape 4 : nettoyage des espaces parasites ---
    # Les champs du corpus contiennent parfois des espaces en début/fin
    for col in df_lines.columns:
        df_lines[col] = df_lines[col].str.strip()
    for col in df_characters.columns:
        df_characters[col] = df_characters[col].str.strip()
    for col in df_movies.columns:
        df_movies[col] = df_movies[col].str.strip()

    # --- Étape 5 : fusion des répliques avec les métadonnées personnages ---
    # On joint sur character_id pour récupérer le genre du personnage
    df_characters_reduit = df_characters[["character_id", "gender"]].drop_duplicates()
    df = df_lines.merge(df_characters_reduit, on="character_id", how="left")

    # --- Étape 6 : fusion avec les métadonnées des films ---
    # On joint sur movie_id pour récupérer le titre et l'année
    df_movies_reduit = df_movies[["movie_id", "movie_title", "movie_year"]].drop_duplicates()
    df = df.merge(df_movies_reduit, on="movie_id", how="left")

    # --- Étape 7 : nettoyage du genre ---
    # Le corpus utilise 'm', 'f', ou '?' pour le genre des personnages
    # On normalise en minuscules et on remplace les valeurs manquantes par '?'
    df["gender"] = df["gender"].str.lower().str.strip()
    df.loc[~df["gender"].isin(["m", "f"]), "gender"] = "?"

    # --- Étape 8 : conversion de l'année en numérique ---
    # Certaines années sont mal formatées — on force la conversion
    df["movie_year"] = pd.to_numeric(df["movie_year"], errors="coerce")

    # --- Étape 9 : sélection et renommage des colonnes finales ---
    df_final = df[["text", "gender", "movie_title", "movie_year"]].copy()
    df_final = df_final.rename(columns={"gender": "character_gender"})

    return df_final


# =============================================================================
# FONCTION 4 : affichage des statistiques du corpus
# =============================================================================

def afficher_stats(df):
    """
    Affiche des statistiques descriptives sur le DataFrame Cornell chargé.

    Utile pour vérifier la qualité des données et comprendre la distribution
    des genres avant l'analyse croisée.

    Paramètre :
      df (DataFrame) : le DataFrame retourné par charger_cornell()
    """
    print("\n" + "=" * 60)
    print("STATISTIQUES DU CORPUS CORNELL MOVIE-DIALOGS")
    print("=" * 60)

    # Nombre total de répliques
    print(f"\nNombre total de répliques : {len(df):,}")

    # Distribution du genre des personnages
    print("\n--- Distribution du genre des personnages ---")
    dist_genre = df["character_gender"].value_counts()
    for genre, count in dist_genre.items():
        pct = count / len(df) * 100
        label = {"m": "Masculin", "f": "Féminin", "?": "Inconnu"}.get(genre, genre)
        print(f"  {label:<12} : {count:>7,} répliques ({pct:.1f}%)")

    # Nombre de films
    nb_films = df["movie_title"].nunique()
    print(f"\nNombre de films : {nb_films}")

    # Plage d'années
    annees_valides = df["movie_year"].dropna()
    if len(annees_valides) > 0:
        print(f"Plage d'années : {int(annees_valides.min())} — {int(annees_valides.max())}")

    # Répliques avec genre connu (m ou f) — c'est la donnée exploitable
    nb_genre_connu = len(df[df["character_gender"].isin(["m", "f"])])
    pct_connu = nb_genre_connu / len(df) * 100
    print(f"\nRépliques avec genre connu (m/f) : {nb_genre_connu:,} ({pct_connu:.1f}%)")

    # Aperçu des premières lignes
    print("\n--- Aperçu (5 premières répliques) ---")
    print(df.head(5).to_string(index=False))


# =============================================================================
# EXÉCUTION PRINCIPALE (si on lance ce fichier directement)
# =============================================================================

if __name__ == "__main__":
    print("=== CHARGEMENT DU CORNELL MOVIE-DIALOGS CORPUS ===\n")

    # Chargement complet (téléchargement + parsing + fusion)
    df_cornell = charger_cornell()

    # Affichage des statistiques
    afficher_stats(df_cornell)
