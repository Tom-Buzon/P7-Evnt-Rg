# Puls-Events — POC RAG (LangChain + Mistral + Faiss)

**Objectif :** démontrer un chatbot capable de répondre à des questions d'utilisateurs sur des événements culturels récents (2024–2025) en s'appuyant sur un pipeline RAG.
Aucune dépendance à Git ou Docker : on utilise **Poetry** pour gérer un **venv**.

## 🚀 Résumé du pipeline
1. **Récupération** des événements via l'API Opendatasoft/OpenAgenda (zone et période configurables).
2. **Nettoyage & normalisation** → CSV → Parquet + champ `text` unifié.
3. **Chunking + embeddings Mistral** → **Index FAISS** persistant.
4. **API FastAPI** avec endpoints `/ask` et `/rebuild`.
5. **Tests** (smoke + structure) et **ébauche d'évaluation** (Ragas).

---

## 📦 Prérequis
- Python ≥ 3.10
- Compte/API Key Mistral (`MISTRAL_API_KEY`)
- Connexion Internet

---

## 🧪 Installation (Poetry + venv)
```bash
# 1) Installer Poetry s'il n'est pas déjà présent
#   Windows (PowerShell):  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
#   Linux/macOS:           curl -sSL https://install.python-poetry.org | python3 -

# 2) Dans ce dossier
poetry install

# 3) Activer l'environnement
poetry shell

# 4) Copier la config
cp .env.example .env
# Éditer .env : votre MISTRAL_API_KEY, filtres REGION/CITY et dates (2024–2025)
```

---

## 🔐 Variables d'environnement (.env)
| Clé | Description |
|-----|-------------|
| `MISTRAL_API_KEY` | Clé API Mistral |
| `MISTRAL_CHAT_MODEL` | Modèle de chat Mistral (ex: `mistral-small-latest`) |
| `MISTRAL_EMBED_MODEL` | Modèle d'embeddings (ex: `mistral-embed`) |
| `DATA_DIR`, `CLEAN_DIR`, `INDEX_DIR` | Dossiers de données et d'index |
| `REGION_FILTER`, `CITY_FILTER` | Filtres géographiques (facultatifs) |
| `DATE_MIN`, `DATE_MAX` | Période (par ex. 2024-01-01 → 2025-12-31) |

> ⚠️ La clé n'est jamais committée. `.env` est ignoré.

---

## 🗂 Structure
```
puls-events-rag-poc/
├─ src/
│  ├─ api/
│  │  └─ app.py                # API FastAPI (/ask, /rebuild)
│  ├─ scripts/
│  │  ├─ fetch_data.py         # 1. Récupérer CSV 2024–2025 (filtré)
│  │  ├─ clean_data.py         # 2. Nettoyage + Parquet
│  │  ├─ build_index.py        # 3. Embedding + FAISS
│  │  ├─ query_rag.py          # 4. Question en CLI (RAG)
│  │  └─ evaluate_rag.py       # 5. Ébauche d'évaluation (Ragas-ready)
│  ├─ openagenda_client.py     # Client Opendatasoft (+ auto-détection du champ date)
│  ├─ config.py                # Chargement .env
│  └─ utils.py                 # Normalisation des colonnes → texte
├─ tests/
│  ├─ test_api_smoke.py        # Démarre l'API et check /docs
│  ├─ test_index.py            # Vérifie présence index
│  └─ dataset.jsonl            # Exemple de dataset d'éval
├─ data/                       # (créé au runtime)
├─ artifacts/                  # (créé au runtime)
├─ scripts/
│  └─ run_api.sh               # Lance l'API (dev)
├─ .env.example
├─ pyproject.toml
└─ README.md
```

---

## 🛠️ Étapes — Scripts distincts et documentés

### 1) Récupérer les données (2024 & 2025)
```bash
poetry run fetch-data
```
- Télécharge via l'API **Opendatasoft/OpenAgenda** (dataset `evenements-publics-openagenda`).
- Détecte automatiquement le champ de date (ex: `date_start`).
- Applique les filtres `REGION_FILTER`, `CITY_FILTER`, `DATE_MIN`, `DATE_MAX`.
- Produit un CSV dans `data/raw/`.

### 2) Nettoyage / Normalisation
```bash
poetry run clean-data
```
- Concatène les CSV raw, supprime doublons simples.
- Harmonise les colonnes clés : `uid,title,description,conditions,keywords,url,city,region,start,end,text`.
- Sauvegarde en **Parquet** → `data/clean/events_clean.parquet`.

### 3) Indexation (Chunks + Embeddings Mistral → FAISS)
```bash
poetry run build-index
```
- Découpage (`RecursiveCharacterTextSplitter`).
- Embeddings **Mistral** (`mistral-embed`).
- Sauvegarde FAISS dans `artifacts/index/`.

### 4) Poser une question en CLI
```bash
poetry run query-rag "Quels concerts à Paris ce week-end ?"
```

### 5) API FastAPI
```bash
# Dev server
./scripts/run_api.sh
# ou
poetry run uvicorn src.api.app:app --reload --port 8000

BONUS : frontend streamlit :
poetry run python -m streamlit run src/streamlit.py
```

Endpoints :
- `POST /ask` → `{ "question": "...", "k": 5 }`
- `POST /rebuild` → reconstruit l'index

Docs Swagger : http://127.0.0.1:8000/docs

### 6) Tests   ______________________________________________________________________________________ Le resste est à dev 🫣🤯😶‍🌫️
```bash
poetry run pytest -q
```

### 7) Évaluation (Ragas — prêt à brancher)
- Ajoutez vos Q/A dans `tests/dataset.jsonl`
- Lancez :
```bash
poetry run evaluate-rag
```

---

## 🧹 Colonnes & nettoyage avant embedding
Le script **`utils.normalize_records_to_df`** sélectionne et renomme dynamiquement les colonnes typiques d'OpenAgenda visibles dans l'interface (cf. capture) : `title/titre`, `description/description_longue`, `detail_des_conditions`, `mots_cles`, `url_canonicale`, `location_city`, `location_region`, `date_start/date_end`, etc.  
Un champ `text` est construit en concaténant **titre + description + conditions + mots-clés** + **métadonnées** (ville, région, dates).  
Cela facilite la vectorisation et la recherche sémantique, tout en conservant **les métadonnées** dans FAISS.

---

## ❗ Points de vigilance
- **Portabilité** : `faiss-cpu` privilégié.
- **Sécurité** : API key dans `.env` uniquement.
- **Perf** : éviter de reconstruire l’index à chaque question (chargement paresseux recommandé si nécessaire).
- **Filtrage** : ajuster `REGION_FILTER`/`CITY_FILTER` pour contrôler le volume.

---

## 📊 Démo attendue
1. Construire l’index (étapes 1–3).
2. Lancer l’API.
3. Tester `POST /ask` avec une question métier (ex. “Que faire à Paris ce week-end ?”).
4. Slides (10–15) basées sur : objectifs, archi, pipeline, résultats, métriques, limites, suites.

Bon POC !
