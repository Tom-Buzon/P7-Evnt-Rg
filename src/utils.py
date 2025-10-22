from typing import Dict, Any
import pandas as pd

CANDIDATE_TITLE = ["title", "titre", "name"]
CANDIDATE_DESC = ["description", "description_longue", "longdescription", "longdescription_fr", "body", "content"]
CANDIDATE_COND = ["detail_des_conditions", "conditions", "conditions_fr", "price", "tarif"]
CANDIDATE_URL = ["url", "url_canonique", "canonical_url", "link"]
CANDIDATE_KEYWORDS = ["mots_cles", "keywords", "keywords_fr", "tags"]
CANDIDATE_CITY = ["location_city", "ville", "city"]
CANDIDATE_REGION = ["location_region", "region"]
CANDIDATE_START = ["firstdate_begin", "date_start", "startdate", "date_debut"]
CANDIDATE_END = ["lastdate_end", "date_end", "enddate", "date_fin"]
CANDIDATE_ID = ["uid", "id", "identifier", "recordid"]

def pick(row: Dict[str, Any], candidates: list[str]) -> Any:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return row[c]
    return None

def normalize_records_to_df(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        fields = r
        title = pick(fields, CANDIDATE_TITLE)
        desc = pick(fields, CANDIDATE_DESC)
        cond = pick(fields, CANDIDATE_COND)
        url = pick(fields, CANDIDATE_URL)
        kw = pick(fields, CANDIDATE_KEYWORDS)
        city = pick(fields, CANDIDATE_CITY)
        region = pick(fields, CANDIDATE_REGION)
        start = pick(fields, CANDIDATE_START)
        end = pick(fields, CANDIDATE_END)
        uid = pick(fields, CANDIDATE_ID) or r.get("id")
        rows.append({
            "uid": uid,
            "title": title,
            "description": desc,
            "conditions": cond,
            "keywords": kw,
            "url": url,
            "city": city,
            "region": region,
            "start": start,
            "end": end
        })
    df = pd.DataFrame(rows).dropna(subset=["title", "description"], how="all")
    # Build a unified text field
    def to_text(x):
        parts = [x.get("title") or "", x.get("description") or "", x.get("conditions") or "", str(x.get("keywords") or "")]
        meta = [f"Ville: {x.get('city') or ''}", f"Région: {x.get('region') or ''}", f"Début: {x.get('start') or ''}", f"Fin: {x.get('end') or ''}"]
        return "\n".join([p for p in parts + meta if p])
    df["text"] = df.apply(lambda row: to_text(row), axis=1)
    return df
