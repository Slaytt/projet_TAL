# =============================================================================
# PRÉTRAITEMENT DES DONNÉES SWDA
# Fichier : src/preprocessing/clean_text.py
# =============================================================================
# Ce fichier contient deux fonctions principales :
#   1. nettoyer_texte()      → nettoie une réplique brute avec spaCy
#   2. preparer_dataset_swda() → applique le nettoyage sur tout le dataset SWDA
#                                et regroupe les 43 labels en 10 macro-classes

import os
import re

import pandas as pd
import spacy
from tqdm import tqdm

# Force le mode offline pour HuggingFace (utilise le cache local)
# Évite que load_dataset() bloque en essayant de vérifier les mises à jour
os.environ["HF_DATASETS_OFFLINE"] = "1"

# =============================================================================
# DICTIONNAIRE DE MAPPING : 43 labels SWDA → 10 macro-classes
# =============================================================================
# Chaque label SWDA (ex: 'sd', 'qy') est associé à une macro-classe.
# Les labels avec None seront considérés comme du BRUIT et supprimés.
# Ce regroupement est crucial pour notre analyse des biais de genre :
#   - POLITESSE (fa=excuse, ft=remerciement) révèle des asymétries de pouvoir
#   - ORDRE (ad=directive) est un marqueur de domination

LABEL_TO_MACROCLASSE = {
    # --- STATEMENT : déclarations factuelles ---
    "sd": "STATEMENT",  # statement-non-opinion : déclaration factuelle
    # --- CONTINUATION : hérite du label précédent dans la conversation ---
    # Le label '+' sera résolu dynamiquement dans preparer_dataset_swda()
    # grâce à un forward-fill par conversation (pas de mapping fixe ici)
    "+": "+",  # placeholder, résolu après
    # --- OPINION : affirmations subjectives ---
    "sv": "OPINION",  # statement-opinion : point de vue personnel
    "bf": "OPINION",  # belief : croyance, bilan personnel
    # --- QUESTION : toutes les formes de questions ---
    "qy": "QUESTION",  # yes-no question : question fermée oui/non
    "qw": "QUESTION",  # wh-question : question ouverte (who, what, where...)
    "qh": "QUESTION",  # rhetorical question : question rhétorique
    "qo": "QUESTION",  # open question : question ouverte sur une opinion
    "qrr": "QUESTION",  # or-clause : question alternative (A ou B ?)
    "qy^d": "QUESTION",  # declarative yes-no : question sous forme déclarative
    "qw^d": "QUESTION",  # declarative wh : question wh sous forme déclarative
    # --- ORDRE : directives, ordres ---
    "ad": "ORDRE",  # action-directive : ordre, demande d'action
    # --- ACCORD : réponses positives ---
    "aa": "ACCORD",  # accept : accord, acceptation
    "aap_am": "ACCORD",  # accept-part / maybe : accord partiel ou hésitant
    "ny": "ACCORD",  # yes-answer : réponse "oui"
    # --- DESACCORD : réponses négatives + plaintes ---
    "no": "DESACCORD",  # no-answer : réponse "non"
    "nn": "DESACCORD",  # no (neutre) : refus neutre
    "ng": "DESACCORD",  # disagree : désaccord explicite
    "ar": "DESACCORD",  # reject : rejet
    "arp_nd": "DESACCORD",  # reject-part / no-defense : désaccord partiel
    "bd": "DESACCORD",  # downplayer : plainte, désapprobation (fusionné ici)
    # --- POLITESSE : régulation sociale (crucial pour les biais de genre !)
    # Ces actes traduisent souvent une asymétrie de pouvoir ou de la soumission
    "fa": "POLITESSE",  # apology : excuse ("I'm sorry", "excuse me")
    "ft": "POLITESSE",  # thanking : remerciement ("thank you", "thanks")
    "fp": "POLITESSE",  # conventional-opening : formule de politesse d'ouverture
    "fc": "POLITESSE",  # conventional-closing : formule de clôture ("bye", "take care")
    # --- BACKCHANNEL : signaux d'écoute active ---
    # Indique que l'interlocuteur écoute sans prendre le tour de parole
    "b": "BACKCHANNEL",  # backchannel : "uh-huh", "right", "yeah"
    "b^m": "BACKCHANNEL",  # backchannel partiel
    "bh": "BACKCHANNEL",  # backchannel sous forme de question ("really?")
    "bk": "BACKCHANNEL",  # acknowledge-answer : accusé de réception
    "ba": "BACKCHANNEL",  # assessment : évaluation/appréciation brève
    "br": "BACKCHANNEL",  # repeat : répétition pour signaler l'écoute
    # --- AUTRE_DIALOGUE : gestion du tour de parole ---
    "^2": "AUTRE_DIALOGUE",  # collaborative completion : fin de phrase de l'autre
    "^g": "AUTRE_DIALOGUE",  # tag-question : tag de fin ("right?", "isn't it?")
    "^h": "AUTRE_DIALOGUE",  # hold avant de prendre le tour
    "^q": "AUTRE_DIALOGUE",  # citation : reprise des mots de l'autre
    "h": "AUTRE_DIALOGUE",  # hedge : hésitation, atténuation
    # --- BRUIT → None : sera filtré et supprimé du dataset ---
    # Ces répliques n'ont pas de signal linguistique exploitable
    # et n'existent pas dans le Cornell Movie-Dialogs Corpus cible
    "%": None,  # fragment abandonné (phrase coupée)
    "x": None,  # non-verbal (bruit, rire...)
    "t1": None,  # self-talk : monologue
    "t3": None,  # joke/anecdote : trop rare pour être appris
    "na": None,  # affirmation négative ambiguë
    'fo_o_fw_"_by_bc': None,  # formules diverses (citations, "parce que"...)
    "oo_co_cc": None,  # offre/option/accord conditionnel : trop rare
}

# =============================================================================
# MOTS GRAMMATICAUX À CONSERVER (non filtrés comme stop words)
# =============================================================================
# Ces mots sont normalement supprimés par spaCy (stop words), mais ils
# portent un signal important pour la classification des dialogue acts.
MOTS_A_GARDER = {
    "do", "does", "did",             # auxiliaires (questions, négations)
    "can", "could", "would",         # modaux
    "will", "should", "might",       # modaux
    "please",                        # signal ORDRE / POLITESSE
    "who", "what", "where",          # mots interrogatifs
    "when", "why", "how",            # mots interrogatifs
    "get", "make",                   # verbes d'action (ORDRE)
    "now",                           # urgence (ORDRE)
    "not", "no",                     # négation (DESACCORD)
    # Verbes courants marqués stop words par spaCy mais utiles pour
    # distinguer les macro-classes (surtout ORDRE et QUESTION)
    "say", "go", "take", "give",     # verbes d'action fréquents
    "put", "see", "keep", "call",    # verbes d'action fréquents
    "show", "move",                  # verbes directifs (ORDRE)
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
         - alphabétiques OU dont le lemme est dans MOTS_A_GARDER
         - non stop words (pas "the", "is", "I"...)
         - de longueur > 1 (évite les lettres isolées résiduelles)

    Paramètres :
      texte (str) : la réplique brute
      nlp         : le modèle spaCy chargé (en_core_web_sm)

    Retourne :
      str : les lemmes utiles séparés par des espaces
            ex: "get car right now not time" pour "Get in the car right now, I don't have time"
    """

    # --- Étape 1 : suppression des marqueurs de disfluence SWDA ---
    # Le corpus SWDA contient des annotations entre accolades {D ...} et
    # crochets [ ... ] qui représentent des disfluences (hésitations, reprises).
    # On les supprime car ils ne font pas partie du contenu linguistique réel.
    texte = re.sub(r"\{[^}]*\}", "", texte)  # supprime tout ce qui est entre { }
    texte = re.sub(r"\[", "", texte)  # supprime les crochets ouvrants
    texte = re.sub(r"\]", "", texte)  # supprime les crochets fermants
    texte = re.sub(r"<<[^>]*>>", "", texte)  # supprime <<long pause>>, <<talking>>...
    texte = re.sub(r"<[^>]*>", "", texte)  # supprime <beep>, <laughter>...

    # --- Étape 1b : mise en minuscule AVANT spaCy ---
    # Nécessaire pour que les mots_a_garder soient reconnus même en début
    # de phrase (ex: "Do" → "do" sera bien identifié comme non-stop word)
    texte = texte.lower()

    # --- Étape 2 : traitement spaCy (tokenisation + lemmatisation) ---
    # nlp(texte) découpe le texte en tokens et calcule le lemme de chacun
    doc = nlp(texte)

    # --- Étape 3 : filtrage des tokens ---
    lemmes_utiles = []
    for token in doc:
        lemme = token.lemma_.lower()
        # On garde un token si :
        #   - il est alphabétique OU son lemme est dans MOTS_A_GARDER
        #     (permet de récupérer "n't" → lemme "not", qui n'est pas alphabétique
        #      à cause de l'apostrophe mais dont le lemme est crucial)
        #   - il n'est pas un stop word
        #   - son lemme fait plus d'1 caractère
        if (token.is_alpha or lemme in MOTS_A_GARDER) and not token.is_stop and len(lemme) > 1:
            lemmes_utiles.append(lemme)

    # On reconstitue une chaîne de caractères à partir des lemmes filtrés
    return " ".join(lemmes_utiles)


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
      3b. Résout les continuations (+) en héritant du label précédent
      4. Supprime les lignes dont la macro-classe est None (BRUIT)
      5. Nettoie chaque réplique avec spaCy (lemmatisation + stop words)
      6. Supprime les répliques vides après nettoyage

    Paramètre :
      dataset : le DatasetDict HuggingFace (chargé avec load_dataset("swda"...))

    Retourne :
      DataFrame Pandas avec les colonnes :
        - 'text'           : la réplique brute originale
        - 'texte_nettoye'  : la réplique nettoyée par spaCy
        - 'macro_classe'   : la macro-classe (STATEMENT, QUESTION, etc.)
    """

    print("Chargement du modèle spaCy...")
    # On charge le modèle d'anglais de spaCy (petit modèle, suffisant pour
    # la lemmatisation et la détection des stop words)
    # disable=["parser", "ner"] accélère spaCy : on n'a besoin que du
    # tagger (pour la lemmatisation), pas de l'analyse syntaxique complète.
    nlp_rapide = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # On retire les mots grammaticaux importants de la liste des stop words
    # + "n't" qui est le token spaCy pour les contractions (don't → "do" + "n't")
    for mot in MOTS_A_GARDER:
        nlp_rapide.vocab[mot].is_stop = False
    nlp_rapide.vocab["n't"].is_stop = False  # contraction → lemme "not"

    print("Conversion du dataset en DataFrame...")
    df = pd.DataFrame(dataset["train"])

    # --- Étape 1 : conversion des labels entiers → noms textuels ---
    # Dans HuggingFace, les labels sont stockés comme des entiers (0, 1, 2...).
    # La feature ClassLabel contient la liste des noms correspondants.
    # On utilise int2str() pour convertir chaque entier en son nom textuel.
    feature_label = dataset["train"].features["damsl_act_tag"]
    df["label_nom"] = df["damsl_act_tag"].apply(lambda i: feature_label.int2str(i))

    # --- Étape 2 : application du mapping → macro-classes ---
    # On remplace chaque nom de label par sa macro-classe.
    # Les labels non présents dans le dictionnaire reçoivent None par défaut.
    df["macro_classe"] = df["label_nom"].map(LABEL_TO_MACROCLASSE)

    # --- Étape 2b : résolution des continuations (+) ---
    # Le label '+' signifie "continuation de la réplique précédente".
    # On hérite la macro-classe de la réplique précédente dans la même conversation.
    # Ex: si la réplique d'avant est QUESTION, la continuation devient QUESTION.
    df = df.sort_values(
        ["conversation_no", "utterance_index", "subutterance_index"]
    ).reset_index(drop=True)
    # On remplace le placeholder '+' par NaN, puis on propage la dernière
    # macro-classe connue vers l'avant (forward-fill) PAR LOCUTEUR.
    # Crucial : si on faisait ffill par conversation seulement, un '+' de
    # speaker A hériterait du backchannel de speaker B au lieu de son propre
    # label précédent. En groupant par (conversation_no, caller), chaque
    # locuteur hérite de son propre historique.
    df["macro_classe"] = df["macro_classe"].replace("+", pd.NA)
    df["macro_classe"] = df.groupby(["conversation_no", "caller"])["macro_classe"].ffill()
    # Si un '+' est la toute première réplique d'un locuteur dans une conversation
    # (rare), il n'a pas de label à hériter → il restera NaN et sera supprimé

    # --- Étape 3 : suppression du BRUIT ---
    # On supprime toutes les lignes dont la macro-classe est None (BRUIT).
    nb_avant = len(df)
    df = df[df["macro_classe"].notna()].copy()
    nb_apres = len(df)
    print(
        f"Lignes supprimées (BRUIT) : {nb_avant - nb_apres} "
        f"({(nb_avant - nb_apres) / nb_avant * 100:.1f}%)"
    )

    # --- Étape 4 : nettoyage du texte avec spaCy ---
    print("Nettoyage des textes avec spaCy...")
    tqdm.pandas(desc="Nettoyage spaCy")
    df["texte_nettoye"] = df["text"].progress_apply(
        lambda t: nettoyer_texte(str(t), nlp_rapide)
    )

    # --- Étape 5 : suppression des répliques vides après nettoyage ---
    # Certaines répliques ne contiennent que des stop words ou de la ponctuation.
    # Après nettoyage, elles deviennent des chaînes vides — on les retire.
    df = df[df["texte_nettoye"].str.strip() != ""].copy()

    print(
        f"\nDataset prêt : {len(df)} répliques | "
        f"{df['macro_classe'].nunique()} macro-classes"
    )
    print("\nDistribution des macro-classes :")
    print(df["macro_classe"].value_counts())

    # On ne retourne que les colonnes utiles pour l'entraînement
    return df[["text", "texte_nettoye", "macro_classe"]].reset_index(drop=True)


# =============================================================================
# TEST RAPIDE (exécuté seulement si on lance ce fichier directement)
# =============================================================================

if __name__ == "__main__":
    from datasets import load_dataset

    print("=== TEST DU PRÉTRAITEMENT (30 premiers exemples) ===\n")
    print("Chargement du dataset SWDA...")
    dataset = load_dataset("swda", trust_remote_code=True)

    # On réduit le dataset à 30 lignes pour tester rapidement
    dataset_reduit = {"train": dataset["train"].select(range(30))}

    df_propre = preparer_dataset_swda(dataset_reduit)

    print("\n--- Résultat ---")
    print(df_propre.to_string())
