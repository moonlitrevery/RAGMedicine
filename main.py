#!/usr/bin/env python3
"""Chat por terminal: memória por sessão, RAG (ChromaDB) e Groq."""

from __future__ import annotations

import os
import sys
import uuid

from dotenv import load_dotenv

from clinic_chat.config import EMBEDDER_PATH
from clinic_chat.memory import MemoryStore
from clinic_chat.rag import chat_with_rag
from clinic_chat.vector_store import get_collection, ingest_from_json


def _ensure_index() -> None:
    if not EMBEDDER_PATH.is_file():
        n = ingest_from_json()
        print(f"Base vetorial inicializada com {n} documentos.\n")
        return
    collection = get_collection()
    if collection.count() == 0:
        n = ingest_from_json()
        print(f"Base vetorial inicializada com {n} documentos.\n")


def main() -> None:
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        print(
            "Erro: defina GROQ_API_KEY (copie .env.example para .env e preencha a chave).",
            file=sys.stderr,
        )
        sys.exit(1)

    _ensure_index()

    user_id = input("user_id (Enter para 'demo'): ").strip() or "demo"
    session_id = input("session_id (Enter para 'sessao1'): ").strip() or "sessao1"

    store = MemoryStore()
    print(
        "\nChat da clínica (RAG + memória). Comandos: /new — nova sessão; /quit — sair.\n"
    )

    while True:
        try:
            line = input(f"[{user_id}@{session_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté logo.")
            break

        if not line:
            continue
        if line in ("/quit", "/exit", "/sair"):
            print("Até logo.")
            break
        if line == "/new":
            store.new_session(user_id, session_id)
            session_id = str(uuid.uuid4())[:8]
            print(f"Memória da sessão anterior limpa. Nova session_id: {session_id}\n")
            continue
        if line.startswith("/"):
            print("Comando desconhecido. Use /new ou /quit.\n")
            continue

        mem = store.get(user_id, session_id)
        history = mem.as_llm_history()

        try:
            answer = chat_with_rag(line, history)
        except Exception as e:
            print(f"Erro ao contactar o modelo: {e}\n", file=sys.stderr)
            continue

        mem.append("user", line)
        mem.append("assistant", answer)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
