# RAGMedicine — Chatbot da clínica (memória + RAG + Groq)

Chat por terminal que combina **memória de conversa** (`user_id` + `session_id`), **ChromaDB** como base vetorial e **Groq** como LLM. O fluxo RAG é: pergunta → recuperação de contexto no Chroma → prompt com contexto → resposta.

## Requisitos

- Python 3.10 ou superior
- Chave API [Groq](https://console.groq.com/keys)

## Instalação

Com [uv](https://docs.astral.sh/uv/) (recomendado):

```bash
cd RAGMedicine
uv sync
cp .env.example .env
# Edite .env e defina GROQ_API_KEY=...
```

Isto cria/usa `.venv` no projeto e instala dependências a partir do `pyproject.toml` (com `uv.lock` para builds reprodutíveis).

Opcional: variável `GROQ_MODEL` (por defeito: `llama-3.3-70b-versatile`).

## Base de conhecimento

15 textos fictícios em `knowledge_base/documents.json` (horários, convénios, exames, regras, contactos). Pode editar ou acrescentar entradas (mantendo o campo `id` único).

## Executar o chat

```bash
uv run python main.py
```

(Alternativa: `source .venv/bin/activate` e depois `python main.py`.)

Indique `user_id` e `session_id`. Na primeira execução, se a coleção Chroma estiver vazia, os documentos são indexados automaticamente para `data/chroma/`. O Chroma pode descarregar na primeira vez o modelo ONNX de embeddings (~80 MB); se falhar por timeout de rede, volte a executar `uv run python main.py` ou `uv run python scripts/ingest.py`.

- **`/new`** — limpa a memória da sessão atual e gera um novo `session_id`.
- **`/quit`** (ou `/exit`, `/sair`) — termina.

Memória de curto prazo: até **20 mensagens** (~10 voltas user/assistant), configurável em `clinic_chat/config.py` (`MEMORY_MAX_MESSAGES`).

## Reindexar manualmente

Após alterar `documents.json`:

```bash
uv run python scripts/ingest.py
```

## Estrutura do código

| Ficheiro | Função |
|----------|--------|
| `main.py` | CLI, comandos `/new` e `/quit` |
| `clinic_chat/memory.py` | Janela de mensagens por `(user_id, session_id)` |
| `clinic_chat/vector_store.py` | ChromaDB persistente, ingestão e `retrieve()` |
| `clinic_chat/rag.py` | Montagem do prompt com contexto + chamada Groq |
| `clinic_chat/config.py` | Caminhos, `RAG_TOP_K`, modelo Groq |

## Alternativa: Qdrant

O enunciado permite Qdrant em alternativa ao Chroma. A lógica de RAG está isolada em `vector_store.py` e `rag.py`; uma variante Qdrant substituiria `retrieve()` e o script de ingestão mantendo a mesma interface.

## Aviso

Este projeto é pedagógico. Não fornece aconselhamento médico real; a “clínica” e os dados são exemplos fictícios.
