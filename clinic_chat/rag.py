from __future__ import annotations

import os

from groq import Groq

from clinic_chat.config import GROQ_MODEL, RAG_TOP_K
from clinic_chat.vector_store import retrieve

SYSTEM_PROMPT = """És o assistente virtual da Clínica Vida Saudável.
Responde de forma clara e empática em português europeu.
Usa APENAS a informação fornecida no contexto abaixo sobre a clínica.
Se a pergunta não puder ser respondida com esse contexto, diz honestamente que não tens essa informação e sugere contactar a secretaria.
Não inventes horários, convénios nem exames que não apareçam no contexto.
Não dês diagnósticos nem aconselhamento médico substituto de um profissional de saúde."""


def build_user_content(question: str, context_chunks: list[str]) -> str:
    context_block = "\n\n---\n\n".join(context_chunks) if context_chunks else "(Sem contexto recuperado da base.)"
    return (
        "Contexto da base de conhecimento da clínica:\n\n"
        f"{context_block}\n\n"
        f"Pergunta do utente: {question}"
    )


def chat_with_rag(
    question: str,
    history: list[dict[str, str]],
    *,
    api_key: str | None = None,
    top_k: int = RAG_TOP_K,
) -> str:
    """
    1. Busca contexto no ChromaDB
    2. Injeta no prompt com histórico de conversa
    3. Chama o LLM Groq
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Defina a variável de ambiente GROQ_API_KEY ou passe api_key=.")

    context_chunks = retrieve(question, n_results=top_k)
    user_content = build_user_content(question, context_chunks)

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    client = Groq(api_key=key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    choice = completion.choices[0]
    return (choice.message.content or "").strip()
