# src/scripts/query_rag.py
from __future__ import annotations

import json, re, unicodedata
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
from mistralai import Mistral
from src.config import settings

# --- FAISS import (compatible anciennes versions)
try:
    from langchain_community.vectorstores import FAISS
except ModuleNotFoundError:
    from langchain.vectorstores import FAISS  # type: ignore

# --- Embeddings: même backend que l’index (Ollama local par défaut)
def _get_ollama_embeddings():
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model="mxbai-embed-large",
            base_url="http://localhost:11434",
        )
    except Exception:  # dernier recours
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            model="mxbai-embed-large",
            base_url="http://localhost:11434",
        )

# -------------------------
# Utils parsing 
# -------------------------
def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

_FR_MONTHS = {
    "janvier":1,"fevrier":2,"février":2,"mars":3,"avril":4,"mai":5,"juin":6,
    "juillet":7,"aout":8,"août":8,"septembre":9,"octobre":10,"novembre":11,"decembre":12,"décembre":12
}

def _parse_full_date_fr(qnorm: str) -> Optional[pd.Timestamp]:
    # 12/09/2024 ou 12-09-2024
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", qnorm)
    if m:
        d, mh, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try: return pd.Timestamp(year=y, month=mh, day=d, tz="UTC")
        except: return None
    # 12 septembre 2024
    m = re.search(r"\b(\d{1,2})\s+([a-zéèêîôûàù]+)\s+(\d{4})\b", qnorm)
    if m:
        d = int(m.group(1)); name = m.group(2); y = int(m.group(3))
        mh = _FR_MONTHS.get(name)
        if mh:
            try: return pd.Timestamp(year=y, month=mh, day=d, tz="UTC")
            except: return None
    # yyyy-mm-dd
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", qnorm)
    if m:
        y, mh, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try: return pd.Timestamp(year=y, month=mh, day=d, tz="UTC")
        except: return None
    return None

def _parse_month_year_fr(qnorm: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    # "septembre 2024"
    m = re.search(r"\b([a-zéèêîôûàù]+)\s+(20\d{2})\b", qnorm)
    if m:
        name = m.group(1); y = int(m.group(2))
        mh = _FR_MONTHS.get(name)
        if mh:
            start = pd.Timestamp(year=y, month=mh, day=1, tz="UTC")
            if mh == 12:
                end = pd.Timestamp(year=y+1, month=1, day=1, tz="UTC") - pd.Timedelta(days=1)
            else:
                end = pd.Timestamp(year=y, month=mh+1, day=1, tz="UTC") - pd.Timedelta(days=1)
            end = end + pd.Timedelta(hours=23, minutes=59, seconds=59)
            return start, end
    # "09/2024"
    m = re.search(r"\b(\d{1,2})[/-](20\d{2})\b", qnorm)
    if m:
        mh, y = int(m.group(1)), int(m.group(2))
        if 1 <= mh <= 12:
            start = pd.Timestamp(year=y, month=mh, day=1, tz="UTC")
            if mh == 12:
                end = pd.Timestamp(year=y+1, month=1, day=1, tz="UTC") - pd.Timedelta(days=1)
            else:
                end = pd.Timestamp(year=y, month=mh+1, day=1, tz="UTC") - pd.Timedelta(days=1)
            end = end + pd.Timedelta(hours=23, minutes=59, seconds=59)
            return start, end
    return None

def _parse_season_range(qnorm: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    # saison + année uniquement, ex: "été 2024"
    m = re.search(r"\b(ete|été|hiver|printemps|automne)\s+(20\d{2})\b", qnorm)
    if not m:
        return None
    season = _norm(m.group(1))
    y = int(m.group(2))
    # fourchettes approximatives
    if season in ("ete", "été"):
        start = pd.Timestamp(year=y, month=6, day=1, tz="UTC")
        end   = pd.Timestamp(year=y, month=8, day=31, hour=23, minute=59, second=59, tz="UTC")
    elif season == "printemps":
        start = pd.Timestamp(year=y, month=3, day=1, tz="UTC")
        end   = pd.Timestamp(year=y, month=5, day=31, hour=23, minute=59, second=59, tz="UTC")
    elif season == "automne":
        start = pd.Timestamp(year=y, month=9, day=1, tz="UTC")
        end   = pd.Timestamp(year=y, month=11, day=30, hour=23, minute=59, second=59, tz="UTC")
    else:  # hiver
        start = pd.Timestamp(year=y, month=12, day=1, tz="UTC")
        end   = pd.Timestamp(year=y+1, month=2, day=28, hour=23, minute=59, second=59, tz="UTC")
    return start, end

def _extract_city_token(query: str) -> Optional[str]:
    m = re.search(r"\b(?:a|à|sur|dans|pres de|près de)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)", query, flags=re.IGNORECASE)
    if m: return _norm(m.group(1))
    caps = re.findall(r"\b([A-ZÀ-Ö][A-Za-zÀ-ÖØ-öø-ÿ\-']+)\b", query)
    return _norm(caps[0]) if caps else None

def _date_hits_range(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp],
                     want_date: Optional[pd.Timestamp],
                     want_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
    if want_date is not None:
        if pd.isna(start): return False
        if pd.notna(end):  return start <= want_date <= end
        return abs((start - want_date).total_seconds()) <= 86400
    if want_range is not None:
        if pd.isna(start): return False
        if pd.notna(end):
            return not (end < want_range[0] or start > want_range[1])
        return want_range[0] <= start <= want_range[1]
    return True

def _sim_from_distance(dist: float) -> float:
    # faiss renvoie souvent une "distance" -> on la convertit en score lisible [0..1[
    try:
        return 1.0 / (1.0 + float(dist))
    except Exception:
        return 0.0

# -------------------------
# Prompt chat
# -------------------------
SYS_PROMPT = (
    "Tu es un assistant amical qui répond aux questions sur des événements en France. "
    "Utilise UNIQUEMENT le contexte fourni (aucune invention). "
    "Quand tu listes plusieurs événements, respecte EXACTEMENT l’ordre des passages dans le contexte : "
    "ils sont déjà triés par pertinence. "
    "Pour chaque événement, fournis au minimum : le titre, la ville, la date/heure (début → fin si disponible) "
    "et le lien (URL). "
    "Si rien n’est pertinent dans le contexte, réponds simplement : « Je ne sais pas. »"
)

# -------------------------
# Main pipeline
# -------------------------
def answer(
    query: str,
    k: int = 5,
    k_base: int = 200,
    use_mmr: bool = True,
    mmr_lambda: float = 0.35,                # valeur par défaut
    mmr_lambda_mult: float | None = None,    # <-- alias accepté depuis l'UI
    mmr_fetch_k: int | None = None,          # déjà ajouté précédemment
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> dict:
    ...
    # --- embeddings + index (inchangé)
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
    vs = FAISS.load_local(str(settings.index_dir), embeddings, allow_dangerous_deserialization=True)

    # --- encodage requête (inchangé)
    qvec = embeddings.embed_query(query)

    # --- retrieval (MMR ou classique) AVEC scores
    docs_with_scores: list[tuple] = []
    if use_mmr:
        # si l'UI fournit mmr_lambda_mult, on le priorise; sinon on garde mmr_lambda
        lambda_mult = mmr_lambda_mult if mmr_lambda_mult is not None else mmr_lambda

        # fetch_k plus large que k_base pour la diversité
        fetch_k_val = mmr_fetch_k if (mmr_fetch_k and mmr_fetch_k > k_base) else max(k_base * 2, 100)

        base_docs = vs.max_marginal_relevance_search_by_vector(
            embedding=qvec,
            k=k_base,
            fetch_k=fetch_k_val,
            lambda_mult=lambda_mult,   
        )

        # Recalcule un score cosinus propre
        import numpy as np
        embs_docs = embeddings.embed_documents([d.page_content for d in base_docs])
        q = np.array(qvec, dtype=float)
        qn = np.linalg.norm(q) + 1e-9
        for pos, (d, dv) in enumerate(zip(base_docs, embs_docs), start=1):
            v = np.array(dv, dtype=float)
            score = float(np.dot(q, v) / (np.linalg.norm(v) * qn + 1e-9))
            docs_with_scores.append((d, score, pos))  # (doc, sim_score, mmr_rank)
    else:
        base = vs.similarity_search_with_score(query, k=k_base)
        for pos, (d, dist) in enumerate(base, start=1):
            sim = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
            docs_with_scores.append((d, sim, pos))

    # --- parsing requête (ville/date) & filtres
    qnorm = _norm(query)
    want_city = _extract_city_token(query)
    want_date = _parse_full_date_fr(qnorm)
    want_range = _parse_month_year_fr(qnorm)
    want_concert = ("concert" in qnorm) or ("musique" in qnorm)

    filtered = []
    for (d, sim, mmr_rank) in docs_with_scores:
        m = d.metadata or {}
        city = _norm(m.get("city"))
        start = pd.to_datetime(m.get("start"), errors="coerce", utc=True)
        end   = pd.to_datetime(m.get("end"), errors="coerce", utc=True)

        ok_city = True if not want_city else (want_city == city or want_city in city)
        ok_date = _date_hits_range(start, end, want_date, want_range)

        ok_topic = True
        if want_concert:
            title = _norm(m.get("title"))
            kws = [ _norm(x) for x in (m.get("keywords") or []) ]
            ok_topic = ("concert" in title) or ("concert" in " ".join(kws)) or ("musique" in title)

        if ok_city and ok_date and ok_topic:
            filtered.append((d, sim, mmr_rank))

    # fallback: si trop strict, relaxe le "topic"
    if not filtered:
        for (d, sim, mmr_rank) in docs_with_scores:
            m = d.metadata or {}
            city = _norm(m.get("city"))
            start = pd.to_datetime(m.get("start"), errors="coerce", utc=True)
            end   = pd.to_datetime(m.get("end"), errors="coerce", utc=True)
            ok_city = True if not want_city else (want_city == city or want_city in city)
            ok_date = _date_hits_range(start, end, want_date, want_range)
            if ok_city and ok_date:
                filtered.append((d, sim, mmr_rank))

    # --- TRI FINAL POUR LA RÉPONSE: sim_score décroissant
    filtered.sort(key=lambda t: t[1], reverse=True)
    chosen = filtered[:k]

    # --- Contexte + ranking (pos_in_context basé sur cet ordre final)
    context_blocks = []
    ranking = []
    for idx, (d, sim, mmr_rank) in enumerate(chosen, start=1):
        m = d.metadata or {}
        header = (
            f"Titre: {m.get('title')} | Ville: {m.get('city')} | Début: {m.get('start')} | "
            f"URL: {m.get('url') or m.get('source') or m.get('canonicalurl')}"
        )
        block = f"{header}\n{d.page_content}"
        context_blocks.append(block)

        ranking.append({
            "pos_in_context": idx,
            "mmr_rank": mmr_rank,
            "sim_score": float(sim),
            "title": m.get("title"),
            "city": m.get("city"),
            "start": m.get("start"),
            "url": m.get("url") or m.get("source") or m.get("canonicalurl"),
            "uid": m.get("uid"),
        })

    # --- Appel Mistral (garde l'ordre du contexte, le prompt l’impose)
    client = Mistral(api_key=settings.mistral_api_key)
    context = "\n\n---\n\n".join(context_blocks)
    resp = client.chat.complete(
        model=getattr(settings, "mistral_chat_model", "mistral-medium-2508"),
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"Contexte (trié par pertinence):\n{context}\n\nQuestion: {query}"},
        ],
    )

    print(
        f"[DEBUG] retrieved(MMR)={len(docs_with_scores)} | "
        f"after_filters={len(filtered)} | used={len(chosen)} | "
        f"city={want_city} | date={want_date} | range={want_range}"
    )

    return {
        "answer": resp.choices[0].message.content,
        "contexts": context_blocks,
        "ranking": ranking,
        "k": k,
    }


def main():
    import sys
    q = " ".join(sys.argv[1:]) or "Quels concerts à Toulouse en juin 2024 ?"
    res = answer(q, k=5, use_mmr=True)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
