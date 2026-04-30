#!/usr/bin/env python3
"""Reindexa a base de conhecimento no ChromaDB (útil após editar documents.json)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinic_chat.vector_store import ingest_from_json  # noqa: E402


def main() -> None:
    n = ingest_from_json()
    print(f"Indexados {n} documentos.")


if __name__ == "__main__":
    main()
