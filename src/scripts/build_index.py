# src/scripts/build_index.py
from __future__ import annotations

import os
import unicodedata
from typing import List
import pandas as pd
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.config import settings

# ==== BACKEND EMBEDDINGS ====
EMBED_BACKEND = "ollama"  # "ollama" ou "mistral"

# Ollama local
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mxbai-embed-large")  # ex: "nomic-embed-text"

# Mistral API
MISTRAL_MODEL = settings.mistral_embed_model

# === Réglages de perf ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2200"))        # ++ pour moins de chunks
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))    # -- overlap
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))        # micro-batch pour embeddings
FAISS_FLUSH = int(os.getenv("FAISS_FLUSH", "1024"))      # nb vecteurs ajoutés à FAISS par flush
MAX_DOCS = int(os.getenv("MAX_DOCS", "0"))               # 0 = pas de limite

def _norm(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def _iso(o) -> str:
    if o is None or (isinstance(o, float) and pd.isna(o)):
        return ""
    try:
        return pd.to_datetime(o, utc=True, errors="coerce").isoformat().replace("+00:00", "Z")
    except Exception:
        return str(o)

def _normalize_keywords(val) -> list[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val)
    parts = []
    for sep in [";", ",", "|"]:
        if sep in s:
            parts = [x.strip() for x in s.split(sep)]
            break
    if not parts:
        parts = [s]
    return [p for p in parts if p]

def _month_to_season(m: int) -> str:
    # saison "calendaire" simple : été = 6..9 (inclus), etc.
    if m in (12, 1, 2):
        return "hiver"
    if m in (3, 4, 5):
        return "printemps"
    if m in (6, 7, 8, 9):
        return "ete"
    return "automne"

def _make_embeddings():
    if EMBED_BACKEND.lower() == "ollama":
        # provider moderne
        try:
            from langchain_ollama import OllamaEmbeddings
        except Exception:
            from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    else:
        from langchain_mistralai import MistralAIEmbeddings
        return MistralAIEmbeddings(api_key=settings.mistral_api_key, model=MISTRAL_MODEL)

def _embed_in_micro_batches(emb, texts: List[str], batch_size: int) -> List[List[float]]:
    """Embeddings avec barre de progression et micro-batching régulier."""
    vectors: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i:i + batch_size]
        vecs = emb.embed_documents(batch)  # liste de vecteurs
        vectors.extend(vecs)
    return vectors

def main():
    # On indexe la version "small" par défaut
    parquet = settings.clean_dir / "events_clean_small.parquet"
    if not parquet.exists():
        parquet = settings.clean_dir / "events_clean.parquet"
    if not parquet.exists():
        raise FileNotFoundError("Missing clean parquet. Run `poetry run clean-data` first.")

    df = pd.read_parquet(parquet)
    if MAX_DOCS > 0:
        df = df.head(MAX_DOCS).copy()

    # Colonnes attendues (depuis le clean)
    # uid, text, title, location_city, location_region, location_postalcode,
    # keywords_list, start, end, canonicalurl, lat, lon
    # On fabrique les champs normalisés utiles à la recherche structurée
    df["city_norm"] = df.get("location_city", "").map(_norm)
    df["region_norm"] = df.get("location_region", "").map(_norm)

    start_ts = pd.to_datetime(df.get("start"), errors="coerce", utc=True)
    df["year"] = start_ts.dt.year.fillna(-1).astype(int)
    df["month"] = start_ts.dt.month.fillna(-1).astype(int)
    df["season"] = start_ts.dt.month.map(lambda m: _month_to_season(int(m)) if pd.notna(m) else "").fillna("")

    # 1) Chunking (rapide, local) — on ajoute des TOKENS META dans le texte indexé
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts: List[str] = []
    metas: List[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        base_text = (row.get("text") or "").strip()
        if not base_text:
            continue

        # keywords
        keywords = row.get("keywords_list")
        if not isinstance(keywords, (list, tuple)):
            keywords = _normalize_keywords(keywords)

        # tokens structurés injectés DANS le texte (stables / génériques)
        # => ils guident la similarité sans “patch” par mot-clé côté requête
        tok = []
        if row.get("city_norm"):
            tok.append(f"city:{row['city_norm']}")
        if row.get("region_norm"):
            tok.append(f"region:{row['region_norm']}")
        if row.get("year", -1) != -1:
            tok.append(f"year:{int(row['year'])}")
        if row.get("month", -1) != -1:
            tok.append(f"month:{int(row['month']):02d}")
            tok.append(f"season:{_month_to_season(int(row['month']))}")
        if keywords:
            tok.extend([f"kw:{_norm(k)}" for k in keywords])

        meta_tokens = " ".join(tok).strip()
        full_text = base_text if not meta_tokens else f"{base_text}\n\n{meta_tokens}"

        chunks = splitter.split_text(full_text)
        if not chunks:
            continue

        meta_base = {
            "uid": row.get("uid"),
            "title": row.get("title") or "",
            "url": row.get("canonicalurl") or "",
            "city": row.get("location_city") or "",
            "city_norm": row.get("city_norm") or "",
            "region": row.get("location_region") or "",
            "region_norm": row.get("region_norm") or "",
            "postalcode": row.get("location_postalcode") or "",
            "start": _iso(row.get("start")),
            "end": _iso(row.get("end")),
            "year": int(row.get("year", -1)),
            "month": int(row.get("month", -1)),
            "season": row.get("season") or "",
            "keywords": keywords or [],
            "source": row.get("canonicalurl") or "",
        }

        for ch in chunks:
            texts.append(ch)
            metas.append(meta_base)

    if not texts:
        raise RuntimeError("No chunks to embed. Check the input parquet.")

    # 2) Embeddings (avec micro-batching + barre de progression)
    embeddings = _make_embeddings()
    vectors = _embed_in_micro_batches(embeddings, texts, EMBED_BATCH)

    # 3) Construction FAISS par flush réguliers
    vs = None
    start = 0
    total = len(vectors)
    while start < total:
        end = min(start + FAISS_FLUSH, total)
        chunk_vecs = vectors[start:end]
        chunk_texts = texts[start:end]
        chunk_metas = metas[start:end]

        if vs is None:
            try:
                vs = FAISS.from_embeddings(
                    list(zip(chunk_texts, chunk_vecs)),
                    embedding=embeddings,
                    metadatas=chunk_metas,
                    normalize_L2=True,
                )
            except TypeError:
                vs = FAISS.from_texts(chunk_texts, embedding=embeddings, metadatas=chunk_metas, normalize_L2=True)
                vs.index.reset()
        else:
            if hasattr(vs, "add_embeddings"):
                vs.add_embeddings(
                    list(zip(chunk_texts, chunk_vecs)),
                    metadatas=chunk_metas,
                    normalize_L2=True,
                )
            elif hasattr(vs, "add_vectors"):
                vs.add_vectors(chunk_vecs, metadatas=chunk_metas)
            else:
                vs.add_texts(chunk_texts, metadatas=chunk_metas, embedding=embeddings, normalize_L2=True)

        print(f"✅ FAISS flush: {end}/{total} vectors")
        start = end

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(settings.index_dir))
    print(f"\n✅ Index FAISS sauvegardé: {settings.index_dir}")
    print(f"Total documents      : {len(df)}")
    print(f"Total chunks         : {len(texts)}")
    print(f"Backend              : {EMBED_BACKEND} | Model: {OLLAMA_MODEL if EMBED_BACKEND=='ollama' else MISTRAL_MODEL}")
    print(f"Chunk size / overlap : {CHUNK_SIZE}/{CHUNK_OVERLAP}")
    print(f"Embed batch / flush  : {EMBED_BATCH}/{FAISS_FLUSH}")

if __name__ == "__main__":
    main()
