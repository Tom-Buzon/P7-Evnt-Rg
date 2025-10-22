# src/scripts/clean_data.py
from __future__ import annotations
import re
import unicodedata
import pandas as pd
from src.config import settings

# --- sélection de régions en DUR (pas de .env) ---
SELECTED_LOCATION = [ "Occitanie"]
#SELECTED_LOCATION = ["Île-de-France", "Occitanie"]

RAW_PARQUET = settings.data_dir / "openagenda_from2024.parquet"
OUT_PARQUET = settings.clean_dir / "events_clean.parquet"
OUT_PARQUET_SMALL = settings.clean_dir / "events_clean_small.parquet"

# --- utils ---
TAG_RE = re.compile(r"<[^>]+>")

def strip_html(s: str | float | None) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    s = TAG_RE.sub(" ", s)                 # remove HTML tags
    s = re.sub(r"\s+", " ", s).strip()     # collapse spaces
    return s

def split_keywords(s: str | float | None) -> list[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    parts = re.split(r"[;,|]", str(s))
    return [p.strip() for p in parts if p and p.strip()]

def coalesce_datetime(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Return the first non-null datetime among the given cols (UTC)."""
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for c in cols:
        if c in df.columns:
            ser = pd.to_datetime(df[c], errors="coerce", utc=True)
            out = out.fillna(ser)
    return out

def norm_str(s: str | float | None) -> str:
    """Normalise (supprime accents, casefold) pour comparer proprement."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # enlève accents
    return s.casefold()

def main():
    if not RAW_PARQUET.exists():
        raise FileNotFoundError(
            f"Parquet introuvable: {RAW_PARQUET}\n"
            "Exécute d'abord: `poetry run python src/scripts/fetch_all_from2024.py`"
        )

    print(f"Lecture: {RAW_PARQUET}")
    df = pd.read_parquet(RAW_PARQUET)

    # --- garder uniquement les colonnes utiles si présentes ---
    wanted = [
        "uid",
        "title_fr", "description_fr", "longdescription_fr", "conditions_fr",
        "keywords_fr",
        "canonicalurl",
        "location_city", "location_region", "location_postalcode", "location_countrycode",
        "location_coordinates.lon", "location_coordinates.lat",
        "firstdate_begin", "lastdate_begin", "firstdate_end", "lastdate_end", "updatedat",
    ]
    available = [c for c in wanted if c in df.columns]
    df = df[available].copy()

    # --- nettoyage texte & construction du champ text ---
    df["title"] = df.get("title_fr", "").apply(strip_html)
    df["desc"] = df.get("description_fr", "").apply(strip_html)
    df["long"] = df.get("longdescription_fr", "").apply(strip_html)
    df["cond"] = df.get("conditions_fr", "").apply(strip_html)

    df["text"] = (df["title"] + ". " + df["desc"] + " " + df["long"] + " " + df["cond"]).str.strip()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

    # --- métadonnées : keywords en liste, géo, etc. ---
    if "keywords_fr" in df.columns:
        df["keywords_list"] = df["keywords_fr"].apply(split_keywords)
    else:
        df["keywords_list"] = [[] for _ in range(len(df))]

    df["lon"] = pd.to_numeric(df.get("location_coordinates.lon", pd.NA), errors="coerce")
    df["lat"] = pd.to_numeric(df.get("location_coordinates.lat", pd.NA), errors="coerce")

    # --- dates robustes (UTC) ---
    df["start"] = coalesce_datetime(df, ["firstdate_begin", "lastdate_begin", "firstdate_end"])
    df["end"]   = coalesce_datetime(df, ["lastdate_end", "firstdate_end", "lastdate_begin"])

    # --- filtrages simples ---
    df = df[df["text"].str.len() >= 10].copy()

    # dédup: même uid + title + start
    dedup_keys = [k for k in ["uid", "title", "start"] if k in df.columns]
    if dedup_keys:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_keys, keep="first").copy()
        print(f"Dédup: {before} -> {len(df)}")

    # --- dataframe final pour l'indexation RAG ---
    final_cols = [
        "uid", "text", "title",
        "location_city", "location_region", "location_postalcode", "location_countrycode",
        "keywords_list", "start", "end", "canonicalurl",
        "lat", "lon",
    ]
    for c in final_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df_final = df[final_cols].copy()

    # Sauvegarde (full)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUT_PARQUET, index=False)

    print("\n=== FULL — Aperçu après nettoyage ===")
    print(df_final.info())
    print(df_final.head(3))
    print(f"\n✅ Sauvegardé (full): {OUT_PARQUET}")

    # --- version "small" filtrée par régions sélectionnées ---
    selected_norm = {norm_str(x) for x in SELECTED_LOCATION}
    if "location_region" in df_final.columns and selected_norm:
        reg_norm = df_final["location_region"].fillna("").apply(norm_str)
        mask = reg_norm.isin(selected_norm)
        df_small = df_final[mask].copy()
    else:
        df_small = df_final.copy()

    df_small.to_parquet(OUT_PARQUET_SMALL, index=False)
    print("\n=== SMALL — Filtre régions ===")
    print(f"SELECTED_LOCATION = {SELECTED_LOCATION}")
    print(df_small["location_region"].value_counts(dropna=False).head(10))
    print(f"✅ Sauvegardé (small): {OUT_PARQUET_SMALL}")

if __name__ == "__main__":
    main()
