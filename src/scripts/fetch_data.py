# src/scripts/fetch_all_from2024.py
import requests
import pandas as pd
from src.config import settings

EXPORT_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "evenements-publics-openagenda/exports/json"
)

def main():
    params = {
        "where": "firstdate_begin >= date'2024-01-01'",
        "lang": "fr",
        "timezone": "Europe/Paris",
    }
    print("Téléchargement en cours…")
    r = requests.get(EXPORT_URL, params=params, timeout=600)
    r.raise_for_status()
    data = r.json()  # liste de dicts

    print(f"Nombre total d'events récupérés : {len(data)}")

    df = pd.json_normalize(data)

    print("\n=== df.info() ===")
    print(df.info())
    print("\n=== Aperçu ===")
    print(df.head())

    out_path = settings.data_dir / "openagenda_from2024.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n✅ Sauvegardé : {out_path}")

if __name__ == "__main__":
    main()
