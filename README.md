# Insurance Claims RAG API

Semantic search over insurance claim outcomes using ChromaDB vector store + FastAPI.

## Run

```bash
docker compose up --build
```

API available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

## Example Queries

### Semantic search over all claims
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "mental health treatment denied", "n_results": 3}'
```

### Filter to denied claims only
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "imaging denied no prior treatment", "n_results": 3, "outcome_filter": "DENIED"}'
```

### Filter to approved claims only
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "failed multiple medications approved", "n_results": 3, "outcome_filter": "APPROVED"}'
```

### Get all denied claims
```bash
curl http://localhost:8000/claims?outcome=DENIED
```

### Get approval statistics
```bash
curl http://localhost:8000/stats
```

### Get specific claim
```bash
curl http://localhost:8000/claims/claim_010
```

## Architecture

```
User Query
    │
    ▼
FastAPI /query endpoint
    │
    ▼
ChromaDB vector store
(sentence-transformer embeddings)
    │
    ▼
Top-k similar claims retrieved
    │
    ▼
RAG summary generated
    │
    ▼
JSON response with ranked results + summary
```

## Extending This

To add an LLM layer (actual RAG generation):
1. Add `openai` or `anthropic` to requirements.txt
2. Pass retrieved claims as context to the LLM
3. Ask the LLM: "Given these similar claims, will this new claim be approved?"

The current version does the retrieval half of RAG.
The generation half would be the LLM synthesizing a prediction from retrieved context.
```
