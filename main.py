from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import chromadb
from chromadb.utils import embedding_functions
import json
import os

# ── Data ──────────────────────────────────────────────────────────────────────

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
        "reason": "Approved - patient meets criteria: HbA1c > 8%, on insulin therapy, physician documented medical necessity.",
        "payer": "Aetna",
        "cost": 1200,
    },
    {
        "id": "claim_003",
        "outcome": "DENIED",
        "procedure": "Bariatric surgery",
        "diagnosis": "Obesity BMI 38",
        "reason": "Denied - BMI below threshold of 40 required without comorbidities. Patient does not have documented hypertension or diabetes.",
        "payer": "UnitedHealth",
        "cost": 25000,
    },
    {
        "id": "claim_004",
        "outcome": "APPROVED",
        "diagnosis": "Major depressive disorder",
        "procedure": "Transcranial magnetic stimulation (TMS)",
        "reason": "Approved - patient failed 3 adequate antidepressant trials, meets payer criteria for treatment-resistant depression.",
        "payer": "Cigna",
        "cost": 8000,
    },
    {
        "id": "claim_005",
        "outcome": "DENIED",
        "diagnosis": "Knee osteoarthritis",
        "procedure": "Platelet-rich plasma (PRP) injection",
        "reason": "Denied - PRP classified as experimental/investigational by payer. Not covered under current benefit plan.",
        "payer": "Medicare",
        "cost": 1800,
    },
    {
        "id": "claim_006",
        "outcome": "APPROVED",
        "diagnosis": "Breast cancer stage II",
        "procedure": "Genetic testing BRCA1/BRCA2",
        "reason": "Approved - family history of breast and ovarian cancer meets NCCN guidelines for genetic counseling and testing.",
        "payer": "BlueCross",
        "cost": 3500,
    },
    {
        "id": "claim_007",
        "outcome": "DENIED",
        "diagnosis": "Insomnia",
        "procedure": "Sleep study polysomnography",
        "reason": "Denied - no documentation of failed behavioral interventions. CBT-I must be attempted before diagnostic sleep study.",
        "payer": "Aetna",
        "cost": 2200,
    },
    {
        "id": "claim_008",
        "outcome": "APPROVED",
        "diagnosis": "Rheumatoid arthritis",
        "procedure": "Adalimumab (Humira) biologic therapy",
        "reason": "Approved - patient failed two DMARDs including methotrexate, documented active disease with elevated CRP and RF.",
        "payer": "Cigna",
        "cost": 20000,
    },
    {
        "id": "claim_009",
        "outcome": "DENIED",
        "diagnosis": "ADHD",
        "procedure": "Neuropsychological testing",
        "reason": "Denied - testing not covered for adult ADHD evaluation. Payer considers clinical interview sufficient for diagnosis.",
        "payer": "UnitedHealth",
        "cost": 3000,
    },
    {
        "id": "claim_010",
        "outcome": "APPROVED",
        "diagnosis": "Bipolar disorder type I",
        "procedure": "Lithium blood level monitoring",
        "reason": "Approved - routine monitoring required for lithium toxicity prevention. Medically necessary for ongoing mood stabilizer management.",
        "payer": "Medicare",
        "cost": 150,
    },
    {
        "id": "claim_011",
        "outcome": "DENIED",
        "diagnosis": "Chronic fatigue syndrome",
        "procedure": "IV immunoglobulin therapy",
        "reason": "Denied - insufficient evidence base for IVIG in CFS. Treatment not supported by current clinical guidelines.",
        "payer": "BlueCross",
        "cost": 12000,
    },
    {
        "id": "claim_012",
        "outcome": "APPROVED",
        "diagnosis": "Epilepsy refractory",
        "procedure": "Vagus nerve stimulator implant",
        "reason": "Approved - patient failed 3 antiepileptic medications, documented refractory epilepsy with frequent seizures affecting daily function.",
        "payer": "Aetna",
        "cost": 35000,
    },
    {
        "id": "claim_013",
        "outcome": "DENIED",
        "diagnosis": "Anxiety disorder",
        "procedure": "Ketamine infusion therapy",
        "reason": "Denied - ketamine for anxiety classified as off-label and experimental. Not covered outside of clinical trial setting.",
        "payer": "Cigna",
        "cost": 4000,
    },
    {
        "id": "claim_014",
        "outcome": "APPROVED",
        "diagnosis": "Atrial fibrillation",
        "procedure": "Catheter ablation",
        "reason": "Approved - patient failed antiarrhythmic drug therapy, symptomatic AF with reduced quality of life, meets cardiology guidelines.",
        "payer": "UnitedHealth",
        "cost": 45000,
    },
    {
        "id": "claim_015",
        "outcome": "DENIED",
        "diagnosis": "Autism spectrum disorder",
        "procedure": "Applied behavior analysis (ABA) therapy adult",
        "reason": "Denied - ABA coverage limited to members under age 21 per benefit plan. Adult ABA not a covered benefit.",
        "payer": "Medicare",
        "cost": 6000,
    },
]


# ── Lifespan ──────────────────────────────────────────────────────────────────

db_client = None
collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, collection

    db_client = chromadb.Client()
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db_client.get_or_create_collection(
        name="insurance_claims",
        embedding_function=ef,
    )

    # Index all claims
    collection.add(
        ids=[c["id"] for c in CLAIMS_DATA],
        documents=[
            f"{c['outcome']} | {c['diagnosis']} | {c['procedure']} | {c['reason']}"
            for c in CLAIMS_DATA
        ],
        metadatas=[
            {k: v for k, v in c.items() if k != "id"}
            for c in CLAIMS_DATA
        ],
    )
    print(f"Indexed {len(CLAIMS_DATA)} claims into vector store")
    yield
    db_client.reset()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Insurance Claims RAG API",
    description="Retrieval-augmented search over insurance claim outcomes",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    n_results: int = 3
    outcome_filter: str | None = None  # "APPROVED" or "DENIED"


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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Insurance Claims RAG API",
        "endpoints": {
            "POST /query": "Semantic search over claims",
            "GET /claims": "List all claims",
            "GET /claims/{claim_id}": "Get a specific claim",
            "GET /stats": "Approval/denial statistics",
        },
    }


@app.get("/health")
def health():
    count = collection.count() if collection else 0
    return {"status": "ok", "claims_indexed": count}


@app.post("/query", response_model=QueryResponse)
def query_claims(request: QueryRequest):
    if not collection:
        raise HTTPException(status_code=503, detail="Vector store not ready")

    where = {"outcome": request.outcome_filter} if request.outcome_filter else None

    results = collection.query(
        query_texts=[request.query],
        n_results=min(request.n_results, len(CLAIMS_DATA)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    claims = []
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    for rank, (meta, claim_id) in enumerate(zip(metadatas, ids), start=1):
        claims.append(
            ClaimResult(
                id=claim_id,
                outcome=meta["outcome"],
                diagnosis=meta["diagnosis"],
                procedure=meta["procedure"],
                reason=meta["reason"],
                payer=meta["payer"],
                cost=meta["cost"],
                relevance_rank=rank,
            )
        )

    # Simple RAG summary
    outcomes = [c.outcome for c in claims]
    approved = outcomes.count("APPROVED")
    denied = outcomes.count("DENIED")

    summary = (
        f"Found {len(claims)} similar claims for '{request.query}'. "
        f"{approved} approved, {denied} denied. "
    )
    if denied > approved:
        top_reason = claims[0].reason
        summary += f"Most common denial pattern: {top_reason[:120]}..."
    else:
        top_reason = claims[0].reason
        summary += f"Top approval rationale: {top_reason[:120]}..."

    return QueryResponse(query=request.query, results=claims, summary=summary)


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
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
    return claim


@app.get("/stats")
def get_stats():
    total = len(CLAIMS_DATA)
    approved = sum(1 for c in CLAIMS_DATA if c["outcome"] == "APPROVED")
    denied = total - approved
    avg_approved_cost = sum(c["cost"] for c in CLAIMS_DATA if c["outcome"] == "APPROVED") / approved
    avg_denied_cost = sum(c["cost"] for c in CLAIMS_DATA if c["outcome"] == "DENIED") / denied

    payer_breakdown = {}
    for c in CLAIMS_DATA:
        p = c["payer"]
        if p not in payer_breakdown:
            payer_breakdown[p] = {"approved": 0, "denied": 0}
        payer_breakdown[p][c["outcome"].lower()] += 1

    return {
        "total_claims": total,
        "approved": approved,
        "denied": denied,
        "approval_rate": round(approved / total * 100, 1),
        "avg_cost_approved": round(avg_approved_cost),
        "avg_cost_denied": round(avg_denied_cost),
        "payer_breakdown": payer_breakdown,
    }
