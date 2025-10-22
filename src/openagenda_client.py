import requests
from typing import Iterable, Dict, Any, List, Set
from urllib.parse import urlencode

BASE = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
DATASET_META = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda"

DATE_CANDIDATES = [
    "firstdate_begin",   # souvent présent
    "date_start",
    "startdate",
    "date",
    "updated_at",
]

def _escape_odsql_str(s: str) -> str:
    """Escape double quotes for ODSQL string literals."""
    return s.replace('"', '\\"')

def _fields_present(limit: int = 1) -> Set[str]:
    """Fetch a single record to detect actual field names on this dataset."""
    params = {"limit": limit}
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    results = js.get("results", [])
    if not results:
        return set()
    return set(results[0].keys())

def detect_date_field() -> str:
    """Pick the best existing date field from candidates by introspection."""
    fields = _fields_present()
    for c in DATE_CANDIDATES:
        if c in fields:
            return c
    # Fallback value used on many OpenAgenda exports
    return "firstdate_begin"

def fetch_events(date_min: str, date_max: str, region: str | None = None, city: str | None = None, batch: int = 1000) -> Iterable[Dict[str, Any]]:
    """Generator fetching events within [date_min, date_max] and optional region/city.
    Uses ODSQL `where` with date literals (date'YYYY-MM-DD') and string equality with double quotes.
    Paginates with `limit`/`offset` (ODSQL constraint: offset+limit <= 10000).
    """
    date_field = detect_date_field()
    offset = 0
    where_clauses = [f"{date_field} >= date'{date_min}'", f"{date_field} <= date'{date_max}'"]
    if region:
        where_clauses.append(f'location_region = "{_escape_odsql_str(region)}"')
    if city:
        where_clauses.append(f'location_city = "{_escape_odsql_str(city)}"')
    where = " AND ".join(where_clauses)

    while True:
        params = {
            "where": where,
            "limit": batch,
            "offset": offset,
            "order_by": f"{date_field}",
        }
        r = requests.get(BASE, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        records: List[Dict[str, Any]] = js.get("results", [])
        if not records:
            break
        for rec in records:
            # Each record is a dict of fields already (v2.1)
            yield rec
        offset += len(records)
        # Hard stop to prevent 10k limit errors (see ODS docs)
        if len(records) < batch or offset + batch > 10000:
            break
