# Projet TAL — Analyse des biais de genre au cinéma

Analyse des dynamiques de pouvoir dans les dialogues de films via la classification des actes de dialogue (Modèle A) et l'extraction de thèmes lexicaux (Modèle B), croisées avec le genre des personnages.

**Corpus :** Cornell Movie-Dialogs Corpus
**Contrainte :** Apprentissage automatique classique uniquement (Scikit-learn) — pas de Deep Learning.

---

## Prérequis

- **Python 3.12** (testé sur 3.12.13 — ne pas utiliser Python 3.13, incompatible avec `datasets==2.18.0`)
- Conda recommandé pour isoler l'environnement

---

## Installation

### 1. Créer l'environnement conda

```bash
conda create -n tal python=3.12
conda activate tal
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger le modèle spaCy

```bash
python -m spacy download en_core_web_sm
```

---

## Lancer le projet

Les étapes 1 et 2 sont nécessaires seulement si les fichiers `.joblib` n'existent pas encore dans `src/model_a/` et `src/model_b/`. Si ils sont déjà présents, passer directement à l'étape 3.

### Étape 1 — Entraîner le Modèle A (classification des actes de dialogue)

Télécharge le dataset SWDA depuis HuggingFace, entraîne un LinearSVC et sauvegarde le modèle.

```bash
python src/model_a/train_classifier.py
```

Produit : `src/model_a/modele_dialogue_acts.joblib`

### Étape 2 — Entraîner le Modèle B (extraction de thèmes LDA)

Charge le corpus Cornell, entraîne une LDA avec 12 thèmes et sauvegarde le modèle.

```bash
python src/model_b/extract_topics.py
```

Produit : `src/model_b/modele_lda.joblib` et `src/model_b/vectoriseur_lda.joblib`

### Étape 3 — Lancer l'analyse croisée

Applique les deux modèles sur le corpus Cornell, croise les résultats avec le genre des personnages et génère les graphiques.

```bash
python src/analysis/merge_results.py
```

Produit dans `resultats/` :
- `resultats_complets.csv` — DataFrame complet (répliques + intentions + thèmes + genre)
- `intentions_par_genre.png` — Distribution des actes de dialogue par genre
- `themes_par_genre.png` — Distribution des thèmes LDA par genre
- `repartition_globale_genre.png` — Répartition globale hommes/femmes

---

## Structure du projet

```
projet_TAL/
├── data/raw/cornell movie-dialogs corpus/   # Corpus brut
├── src/
│   ├── preprocessing/
│   │   ├── load_cornell.py      # Chargement et parsing du corpus Cornell
│   │   └── clean_text.py        # Nettoyage/lemmatisation via spaCy
│   ├── model_a/
│   │   ├── train_classifier.py  # Entraînement LinearSVC sur SWDA
│   │   └── predict_intent.py    # Prédiction des actes de dialogue
│   ├── model_b/
│   │   └── extract_topics.py    # Entraînement LDA + assignation de thèmes
│   └── analysis/
│       └── merge_results.py     # Analyse croisée + visualisations
├── resultats/                   # Graphiques et CSV générés
└── requirements.txt
```
