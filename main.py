from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from anthropic import Anthropic

load_dotenv()
anthropic_client = Anthropic()

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

CLAIMS_DATA = [
    {
        "id": "claim_001",
        "outcome": "DENIED",
        "diagnosis": "Lower back pain",
        "procedure": "MRI lumbar spine",
        "reason": "Denied - no prior conservative treatment documented. Patient must complete 6 weeks of physical therapy before advanced imaging is approved.",
        "payer": "BlueCross",
        "cost": 2400,
    },
    {
        "id": "claim_002",
        "outcome": "APPROVED",
        "diagnosis": "Type 2 diabetes",
        "procedure": "Continuous glucose monitor",
        "reason": "Approved - patient meets criteria: HbA1c > 8%, insulin therapy, physician documented medical necessity.",
        "payer": "Aetna",
        "cost": 1200,
    },
    # keep rest unchanged...
]

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

db_client = None
collection = None


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN (VECTOR DB INIT)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, collection

    db_client = chromadb.Client()
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = db_client.get_or_create_collection(
        name="insurance_claims",
        embedding_function=embedding_fn,
    )

    documents = [
        f"{c['outcome']} | {c['diagnosis']} | {c['procedure']} | {c['reason']}"
        for c in CLAIMS_DATA
    ]

    collection.add(
        ids=[c["id"] for c in CLAIMS_DATA],
        documents=documents,
        metadatas=[{k: v for k, v in c.items() if k != "id"} for c in CLAIMS_DATA],
    )

    print(f"Indexed {len(CLAIMS_DATA)} claims into vector DB")

    yield

    db_client.reset()


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Insurance Claims RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    n_results: int = 3
    outcome_filter: str | None = None


class ClaimResult(BaseModel):
    id: str
    outcome: str
    diagnosis: str
    procedure: str
    reason: str
    payer: str
    cost: int
    relevance_rank: int


class QueryResponse(BaseModel):
    query: str
    results: list[ClaimResult]
    summary: str


# ─────────────────────────────────────────────────────────────────────────────
# RAG CORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_context(metadatas, ids) -> str:
    """Step 1: convert retrieved docs into LLM-ready context"""
    chunks = []

    for cid, m in zip(ids, metadatas):
        chunks.append(
            f"""
Claim ID: {cid}
Outcome: {m["outcome"]}
Diagnosis: {m["diagnosis"]}
Procedure: {m["procedure"]}
Payer: {m["payer"]}
Cost: ${m["cost"]}
Reason: {m["reason"]}
"""
        )

    return "\n---\n".join(chunks)


def generate_response(query: str, context: str) -> str:
    """Step 2: LLM generation grounded in retrieved claims."""

    if not context:
        return f"No similar insurance claims found for: {query}"

    message = anthropic_client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=(
            "You are an insurance claims analyst. Answer the user's question "
            "using ONLY the retrieved past claims provided as context. "
            "Identify approval and denial patterns, cite specific claim IDs, "
            "and be concise."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Retrieved past claims:\n{context}\n\n"
                f"Question: {query}"
            ),
        }],
    )

    return next(block.text for block in message.content if block.type == "text")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Insurance Claims RAG API"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "claims_indexed": collection.count() if collection else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RAG ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_claims(request: QueryRequest):
    if not collection:
        raise HTTPException(status_code=503, detail="Vector DB not ready")

    where = (
        {"outcome": request.outcome_filter.upper()}
        if request.outcome_filter
        else None
    )

    results = collection.query(
        query_texts=[request.query],
        n_results=request.n_results,
        where=where,
        include=["metadatas"],
    )

    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    # ── build structured results
    claims: list[ClaimResult] = []

    for rank, (cid, meta) in enumerate(zip(ids, metadatas), start=1):
        claims.append(
            ClaimResult(
                id=cid,
                outcome=meta["outcome"],
                diagnosis=meta["diagnosis"],
                procedure=meta["procedure"],
                reason=meta["reason"],
                payer=meta["payer"],
                cost=meta["cost"],
                relevance_rank=rank,
            )
        )

    # ── RAG STEP 1: context (ready for LLM)
    context = build_context(metadatas, ids)

    # ── RAG STEP 2: generation
    summary = generate_response(request.query, context)

    return QueryResponse(
        query=request.query,
        results=claims,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BASIC DATA ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/claims")
def list_claims(outcome: str | None = None):
    data = CLAIMS_DATA

    if outcome:
        data = [c for c in data if c["outcome"] == outcome.upper()]

    return {"total": len(data), "claims": data}


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str):
    claim = next((c for c in CLAIMS_DATA if c["id"] == claim_id), None)

    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    return claim


@app.get("/stats")
def get_stats():
    total = len(CLAIMS_DATA)
    approved = sum(1 for c in CLAIMS_DATA if c["outcome"] == "APPROVED")
    denied = total - approved

    avg = lambda status: (
        sum(c["cost"] for c in CLAIMS_DATA if c["outcome"] == status)
        / max(1, (approved if status == "APPROVED" else denied))
    )

    payer_breakdown = {}

    for c in CLAIMS_DATA:
        payer_breakdown.setdefault(c["payer"], {"approved": 0, "denied": 0})
        payer_breakdown[c["payer"]][c["outcome"].lower()] += 1

    return {
        "total_claims": total,
        "approved": approved,
        "denied": denied,
        "approval_rate": round(approved / total * 100, 1),
        "avg_cost_approved": round(avg("APPROVED")),
        "avg_cost_denied": round(avg("DENIED")),
        "payer_breakdown": payer_breakdown,
    }