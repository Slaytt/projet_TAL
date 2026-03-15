# =============================================================================
# PRÉTRAITEMENT DES DONNÉES SWDA
# Fichier : src/preprocessing/clean_text.py
# =============================================================================
# Ce fichier contient deux fonctions principales :
#   1. nettoyer_texte()      → nettoie une réplique brute avec spaCy
#   2. preparer_dataset_swda() → applique le nettoyage sur tout le dataset SWDA
#                                et regroupe les 43 labels en 11 macro-classes

import re
import pandas as pd
import spacy

# =============================================================================
# DICTIONNAIRE DE MAPPING : 43 labels SWDA → 11 macro-classes
# =============================================================================
# Chaque label SWDA (ex: 'sd', 'qy') est associé à une macro-classe.
# Les labels avec None seront considérés comme du BRUIT et supprimés.
# Ce regroupement est crucial pour notre analyse des biais de genre :
#   - POLITESSE (fa=excuse, ft=remerciement) révèle des asymétries de pouvoir
#   - ORDRE (ad=directive) est un marqueur de domination
#   - PLAINTE (bd) peut être genrée

LABEL_TO_MACROCLASSE = {
    # --- STATEMENT : déclarations factuelles + continuations ---
    'sd': 'STATEMENT',  # statement-non-opinion : déclaration factuelle
    '+': 'STATEMENT',   # continuation : suite de la réplique précédente

    # --- OPINION : affirmations subjectives ---
    'sv': 'OPINION',    # statement-opinion : point de vue personnel
    'bf': 'OPINION',    # belief : croyance, bilan personnel

    # --- QUESTION : toutes les formes de questions ---
    'qy': 'QUESTION',   # yes-no question : question fermée oui/non
    'qw': 'QUESTION',   # wh-question : question ouverte (who, what, where...)
    'qh': 'QUESTION',   # rhetorical question : question rhétorique
    'qo': 'QUESTION',   # open question : question ouverte sur une opinion
    'qrr': 'QUESTION',  # or-clause : question alternative (A ou B ?)
    'qy^d': 'QUESTION', # declarative yes-no : question sous forme déclarative
    'qw^d': 'QUESTION', # declarative wh : question wh sous forme déclarative

    # --- ORDRE : directives, ordres ---
    'ad': 'ORDRE',      # action-directive : ordre, demande d'action

    # --- ACCORD : réponses positives ---
    'aa': 'ACCORD',     # accept : accord, acceptation
    'aap_am': 'ACCORD', # accept-part / maybe : accord partiel ou hésitant
    'ny': 'ACCORD',     # yes-answer : réponse "oui"

    # --- DESACCORD : réponses négatives ---
    'no': 'DESACCORD',  # no-answer : réponse "non"
    'nn': 'DESACCORD',  # no (neutre) : refus neutre
    'ng': 'DESACCORD',  # disagree : désaccord explicite
    'ar': 'DESACCORD',  # reject : rejet
    'arp_nd': 'DESACCORD', # reject-part / no-defense : désaccord partiel

    # --- POLITESSE : régulation sociale (crucial pour les biais de genre !)
    # Ces actes traduisent souvent une asymétrie de pouvoir ou de la soumission
    'fa': 'POLITESSE',  # apology : excuse ("I'm sorry", "excuse me")
    'ft': 'POLITESSE',  # thanking : remerciement ("thank you", "thanks")
    'fp': 'POLITESSE',  # conventional-opening : formule de politesse d'ouverture
    'fc': 'POLITESSE',  # conventional-closing : formule de clôture ("bye", "take care")

    # --- BACKCHANNEL : signaux d'écoute active ---
    # Indique que l'interlocuteur écoute sans prendre le tour de parole
    'b': 'BACKCHANNEL',   # backchannel : "uh-huh", "right", "yeah"
    'b^m': 'BACKCHANNEL', # backchannel partiel
    'bh': 'BACKCHANNEL',  # backchannel sous forme de question ("really?")
    'bk': 'BACKCHANNEL',  # acknowledge-answer : accusé de réception
    'ba': 'BACKCHANNEL',  # assessment : évaluation/appréciation brève
    'br': 'BACKCHANNEL',  # repeat : répétition pour signaler l'écoute

    # --- AUTRE_DIALOGUE : gestion du tour de parole ---
    '^2': 'AUTRE_DIALOGUE',  # collaborative completion : fin de phrase de l'autre
    '^g': 'AUTRE_DIALOGUE',  # tag-question : tag de fin ("right?", "isn't it?")
    '^h': 'AUTRE_DIALOGUE',  # hold avant de prendre le tour
    '^q': 'AUTRE_DIALOGUE',  # citation : reprise des mots de l'autre
    'h': 'AUTRE_DIALOGUE',   # hedge : hésitation, atténuation

    # --- PLAINTE : désapprobation, plainte ---
    'bd': 'PLAINTE',     # downplayer : plainte, désapprobation

    # --- BRUIT → None : sera filtré et supprimé du dataset ---
    # Ces répliques n'ont pas de signal linguistique exploitable
    # et n'existent pas dans le Cornell Movie-Dialogs Corpus cible
    '%': None,                  # fragment abandonné (phrase coupée)
    'x': None,                  # non-verbal (bruit, rire...)
    't1': None,                 # self-talk : monologue
    't3': None,                 # joke/anecdote : trop rare pour être appris
    'na': None,                 # affirmation négative ambiguë
    'fo_o_fw_"_by_bc': None,    # formules diverses (citations, "parce que"...)
    'oo_co_cc': None,           # offre/option/accord conditionnel : trop rare
}


# =============================================================================
# FONCTION 1 : nettoyage d'une réplique avec spaCy
# =============================================================================

def nettoyer_texte(texte, nlp):
    """
    Nettoie une réplique brute et retourne une chaîne de lemmes utiles.

    Étapes :
      1. Suppression des marqueurs de disfluence SWDA entre { } et [ ]
         (ces symboles annotent des hésitations dans le corpus, ex: "{D So, }")
      2. Tokenisation + lemmatisation par spaCy
      3. Filtrage : on garde uniquement les mots qui sont :
         - alphabétiques (pas de chiffres ni de ponctuation)
         - non stop words (pas "the", "is", "I"...)
         - de longueur > 1 (évite les lettres isolées résiduelles)

    Paramètres :
      texte (str) : la réplique brute
      nlp         : le modèle spaCy chargé (en_core_web_sm)

    Retourne :
      str : les lemmes utiles séparés par des espaces
            ex: "get car right time" pour "Get in the car right now, I don't have time"
    """

    # --- Étape 1 : suppression des marqueurs de disfluence SWDA ---
    # Le corpus SWDA contient des annotations entre accolades {D ...} et
    # crochets [ ... ] qui représentent des disfluences (hésitations, reprises).
    # On les supprime car ils ne font pas partie du contenu linguistique réel.
    texte = re.sub(r'\{[^}]*\}', '', texte)  # supprime tout ce qui est entre { }
    texte = re.sub(r'\[', '', texte)          # supprime les crochets ouvrants
    texte = re.sub(r'\]', '', texte)          # supprime les crochets fermants

    # --- Étape 2 : traitement spaCy (tokenisation + lemmatisation) ---
    # nlp(texte) découpe le texte en tokens et calcule le lemme de chacun
    doc = nlp(texte)

    # --- Étape 3 : filtrage des tokens ---
    lemmes_utiles = []
    for token in doc:
        # token.is_alpha  : True si le token ne contient que des lettres
        # token.is_stop   : True si c'est un mot vide ("the", "is", "I"...)
        # token.lemma_    : la forme de base du mot ("running" → "run")
        if token.is_alpha and not token.is_stop and len(token.lemma_) > 1:
            lemmes_utiles.append(token.lemma_.lower())

    # On reconstitue une chaîne de caractères à partir des lemmes filtrés
    return ' '.join(lemmes_utiles)


# =============================================================================
# FONCTION 2 : préparation complète du dataset SWDA
# =============================================================================

def preparer_dataset_swda(dataset):
    """
    Prend le dataset SWDA brut (chargé depuis HuggingFace) et retourne
    un DataFrame Pandas propre, prêt pour l'entraînement du Modèle A.

    Ce que fait cette fonction :
      1. Convertit le split 'train' en DataFrame Pandas
      2. Récupère le nom textuel du label (ex: 4 → 'sd') grâce au ClassLabel
      3. Applique le mapping LABEL_TO_MACROCLASSE pour obtenir les macro-classes
      4. Supprime les lignes dont la macro-classe est None (BRUIT)
      5. Nettoie chaque réplique avec spaCy (lemmatisation + stop words)
      6. Supprime les répliques vides après nettoyage

    Paramètre :
      dataset : le DatasetDict HuggingFace (chargé avec load_dataset("swda"...))

    Retourne :
      DataFrame Pandas avec les colonnes :
        - 'texte_nettoye'  : la réplique nettoyée par spaCy
        - 'macro_classe'   : la macro-classe (STATEMENT, QUESTION, etc.)
    """

    print("Chargement du modèle spaCy...")
    # On charge le modèle d'anglais de spaCy (petit modèle, suffisant pour
    # la lemmatisation et la détection des stop words)
    nlp = spacy.load("en_core_web_sm")

    print("Conversion du dataset en DataFrame...")
    df = pd.DataFrame(dataset["train"])

    # --- Étape 1 : conversion des labels entiers → noms textuels ---
    # Dans HuggingFace, les labels sont stockés comme des entiers (0, 1, 2...).
    # La feature ClassLabel contient la liste des noms correspondants.
    # On utilise int2str() pour convertir chaque entier en son nom textuel.
    feature_label = dataset["train"].features["damsl_act_tag"]
    df["label_nom"] = df["damsl_act_tag"].apply(
        lambda i: feature_label.int2str(i)
    )

    # --- Étape 2 : application du mapping → macro-classes ---
    # On remplace chaque nom de label par sa macro-classe.
    # Les labels non présents dans le dictionnaire reçoivent None par défaut.
    df["macro_classe"] = df["label_nom"].map(LABEL_TO_MACROCLASSE)

    # --- Étape 3 : suppression du BRUIT ---
    # On supprime toutes les lignes dont la macro-classe est None (BRUIT).
    nb_avant = len(df)
    df = df[df["macro_classe"].notna()].copy()
    nb_apres = len(df)
    print(f"Lignes supprimées (BRUIT) : {nb_avant - nb_apres} "
          f"({(nb_avant - nb_apres) / nb_avant * 100:.1f}%)")

    # --- Étape 4 : nettoyage du texte avec spaCy ---
    # On applique nettoyer_texte() sur chaque réplique.
    # disable=["parser", "ner"] accélère spaCy : on n'a besoin que du
    # tagger (pour la lemmatisation), pas de l'analyse syntaxique complète.
    print("Nettoyage des textes avec spaCy (peut prendre quelques minutes)...")
    nlp_rapide = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    df["texte_nettoye"] = df["text"].apply(
        lambda t: nettoyer_texte(str(t), nlp_rapide)
    )

    # --- Étape 5 : suppression des répliques vides après nettoyage ---
    # Certaines répliques ne contiennent que des stop words ou de la ponctuation.
    # Après nettoyage, elles deviennent des chaînes vides — on les retire.
    df = df[df["texte_nettoye"].str.strip() != ""].copy()

    print(f"\nDataset prêt : {len(df)} répliques | "
          f"{df['macro_classe'].nunique()} macro-classes")
    print("\nDistribution des macro-classes :")
    print(df["macro_classe"].value_counts())

    # On ne retourne que les deux colonnes utiles pour l'entraînement
    return df[["texte_nettoye", "macro_classe"]].reset_index(drop=True)


# =============================================================================
# TEST RAPIDE (exécuté seulement si on lance ce fichier directement)
# =============================================================================

if __name__ == "__main__":
    from datasets import load_dataset

    print("=== TEST DU PRÉTRAITEMENT ===\n")
    dataset = load_dataset("swda", trust_remote_code=True)
    df_propre = preparer_dataset_swda(dataset)

    print("\n--- Aperçu des 10 premières lignes ---")
    print(df_propre.head(10))
