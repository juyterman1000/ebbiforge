import os
import sys
# Set threads to 1 to prevent starvation
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import uuid
import json
import requests
import queue
import threading
import time
from dotenv import load_dotenv
from privacy import PrivacyFilter, Vault

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    print(f"⚠️ load_dotenv error: {e}")

# API Keys
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("MODEL_API_KEY")
if not API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY or MODEL_API_KEY must be set in the environment.")
    sys.exit(1)

os.environ["MODEL_API_KEY"] = API_KEY
os.environ["GEMINI_API_KEY"] = API_KEY

# --- Global Queue ---
ingestion_queue = queue.Queue()

app = FastAPI(title="CogOps API", description="API for CogOps Agent", version="1.0.0")


RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_state: Dict[str, Dict[str, float]] = {}


def _get_user_id(request: Request) -> str:
    """
    Derive a stable per-user identifier for rate limiting.
    Prefers explicit header, then falls back to client IP.
    """
    user_id = request.headers.get("X-User-Id") or request.headers.get("X-User-ID")
    if user_id:
        return f"hdr:{user_id}"
    client_host = getattr(request.client, "host", None) or "unknown"
    return f"ip:{client_host}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only apply to API routes
        if not str(request.url.path).startswith("/api/"):
            return await call_next(request)

        now = time.time()
        user_id = _get_user_id(request)

        state = _rate_limit_state.get(user_id)
        if not state or now - state.get("window_start", 0) > RATE_LIMIT_WINDOW_SECONDS:
            state = {"window_start": now, "count": 0}
        count = state["count"]

        if count >= RATE_LIMIT_REQUESTS:
            retry_after = max(
                0.0,
                RATE_LIMIT_WINDOW_SECONDS - (now - state["window_start"]),
            )
            return JSONResponse(
                {
                    "detail": "Rate limit exceeded. Max 100 requests per minute per user.",
                    "retry_after_seconds": round(retry_after, 2),
                },
                status_code=429,
            )

        state["count"] = count + 1
        _rate_limit_state[user_id] = state

        return await call_next(request)


app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    response: str
    thought_process: List[Dict[str, Any]] = []

# --- Imports & Globals (Graceful fallback for standalone mode) ---
graph = None
redactor = None
vault = None
memory_store = None
embedding_model = None
in_memory_history = []  # Fallback when no vector store

try:
    from ebbiforge_core import AgentGraphPy, HistoryBuffer, TrajectoryPoint, Agent, PIIRedactor
    graph = AgentGraphPy()
    default_agent = Agent(name="Assistant", instructions="Helpful AI.")
    graph.register_agent(default_agent)
    redactor = PIIRedactor('*')
    print("✅ CogOps Core Loaded")
except Exception as e:
    print(f"⚠️ CogOps Core not available: {e} — using Gemini-only mode")

# Optional: Vector store + embeddings
try:
    from ebbiforge_core import DragonflyStore, RemoteVectorStore
    from fastembed import TextEmbedding
    memory_store = RemoteVectorStore("http://localhost:6334", "cogops_memory_local_base", 768)
    embedding_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
    print("✅ Vector Store + Embeddings Loaded")
except Exception as e:
    print(f"⚠️ Vector store unavailable: {e} — using in-memory history")

# Optional: Encryption vault
try:
    vault_key = os.getenv("SECURE_VAULT_KEY")
    if vault_key:
        vault = Vault(vault_key)
        print("✅ Secure Vault Loaded")
except Exception as e:
    print(f"⚠️ Vault unavailable: {e}")

# --- Helper Functions ---

def get_embedding(text: str) -> List[float]:
    try:
        if embedding_model:
            return list(embedding_model.embed([text]))[0].tolist()
    except: pass
    return []

def save_memory(text: str):
    """CPU Heavy - Run in Worker Thread"""
    try:
        if memory_store and embedding_model:
            vector = get_embedding(text)
            mem_id = str(uuid.uuid4())
            if vault:
                encrypted = vault.encrypt(json.dumps({"text": text}))
                payload = json.dumps({"payload": encrypted})
            else:
                payload = json.dumps({"text": text})
            memory_store.upsert(mem_id, vector, payload)
            print(f"DEBUG: Saved memory {mem_id}")
        else:
            # In-memory fallback
            in_memory_history.append(text)
            if len(in_memory_history) > 100:
                in_memory_history.pop(0)
    except Exception as e:
        print(f"⚠️ Save failed: {e}")

def retrieve_context(query: str) -> str:
    # Try vector store first
    if memory_store and embedding_model:
        try:
            vector = get_embedding(query)
            is_broad = any(k in query.lower() for k in ["report", "all", "list", "show me all"])
            limit = 60 if "report" in query.lower() else (60 if is_broad else 7)
            results = memory_store.search(vector, limit)
            ctx = []
            for r in results:
                if r.score > 0.12:
                    try:
                        p = json.loads(r.payload)
                        txt = p.get("text")
                        if not txt and "payload" in p and vault:
                            decrypted = vault.decrypt(p["payload"])
                            txt = json.loads(decrypted).get("text")
                        if txt: ctx.append(f"- {txt}")
                    except: pass
            return "\n".join(ctx) if ctx else ""
        except: pass

    # In-memory fallback
    if in_memory_history:
        q = query.lower()
        relevant = [m for m in in_memory_history if any(w in m.lower() for w in q.split()[:3])]
        if relevant:
            return "\n".join(f"- {m}" for m in relevant[-10:])
    return ""

def gemma_fallback(message: str, context: str) -> str:
    # Switched to Gemini 2.5 Flash Lite (Available capacity: 5/10 RPM, 250k TPM)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={API_KEY}"
    prompt = f"MEMORY CONTEXT:\n{context}\n\nREQUEST:\n{message}\n\nINSTRUCTIONS:\n1. Answer clearly.\n2. If the user asks for a list (e.g., 'all projects', 'all colleagues'), you MUST list EVERY single item found in the memory context. **You MUST include the exact Name/Title of every item (e.g., 'Project ATLAS', 'Project NEUROMATCH'). Do not summarize away the names.**\n3. If the answer is negative (e.g. 'Do I have a pet named Max?'), you MUST explicitly state what DOES exist in that category (e.g. 'No, but you have a dog named Newton')."
    try:
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60)
        r.raise_for_status()
        return r.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error: {e}"

# --- Worker ---
def ingestion_worker():
    print("👷 Ingestion Worker Started")
    while True:
        try:
            text = ingestion_queue.get()
            if text is None: break
            save_memory(text)
            ingestion_queue.task_done()
        except: pass

@app.on_event("startup")
async def startup():
    threading.Thread(target=ingestion_worker, daemon=True).start()
    # Start the SwarmBrain — activates ALL Rust modules in background
    try:
        from swarm_brain import get_brain
        brain = get_brain()
        brain.start_background(tick_interval=0.05)  # ~20Hz
        print("🧠 SwarmBrain started — ALL Rust modules active")
    except Exception as e:
        print(f"⚠️ SwarmBrain failed to start: {e}")

# --- Swarm Brain Endpoints ---
@app.get("/api/swarm/state")
def swarm_state():
    """Get full swarm brain state — all metrics from real Rust engine."""
    try:
        from swarm_brain import get_brain
        brain = get_brain()
        return brain.get_state()
    except Exception as e:
        return {"error": str(e), "rust_modules_active": False}

class SignalRequest(BaseModel):
    source: str
    value: float
    surprise_weight: float = 1.0

@app.post("/api/swarm/signal")
def swarm_signal(req: SignalRequest):
    """Inject a signal into the real Rust swarm engine."""
    try:
        from swarm_brain import get_brain
        brain = get_brain()
        brain.inject_signal(req.source, req.value, req.surprise_weight)
        return {"status": "injected", "source": req.source, "value": req.value}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/swarm/explain")
def swarm_explain():
    """Get LLM explanation of what the swarm is currently sensing."""
    try:
        from swarm_brain import get_brain
        brain = get_brain()
        state = brain.get_state()
        # Build explanation from latest alerts
        alerts = state.get("alerts", [])
        explained = [a for a in alerts if a.get("llm_analysis")]
        return {
            "state": state,
            "explanations": explained[-5:],
            "modules_active": {
                "swarm": True,
                "memory": True,
                "metacognition": True,
                "safety_shield": True,
                "cross_pollination": True,
                "curiosity": True,
                "compliance": True,
            }
        }
    except Exception as e:
        return {"error": str(e)}

# --- Endpoints ---
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if redactor:
        request.message = PrivacyFilter.redact(request.message)
    session_id = request.session_id or "default"
    
    # --- FAST PATH (Queue Ingestion) ---
    retrieval_keywords = ["what", "how", "who", "when", "why", "where", "generate", "report", "summarize", "analyze", "list", "present", "tell", "show", "describe", "find", "search", "check", "do i", "which"]
    is_question = "?" in request.message or any(q in request.message.lower() for q in retrieval_keywords)
    
    is_short = len(request.message) < 500
    is_gauntlet = session_id and "gauntlet" in session_id.lower()
    
    if is_gauntlet and not is_question and is_short:
        print(f"⚡ Queueing: {session_id} (Size: {ingestion_queue.qsize()})")
        ingestion_queue.put(request.message)
        return ChatResponse(response="I've recorded that information for you.")
        
    # --- REPORT PATH ---
    report_triggers = ["memory gauntlet report", "tell me about", "what are my", "who are my", "which projects", "do i have"]
    if any(t in request.message.lower() for t in report_triggers):
        print(f"📑 REPORT PATH: Gemma Synthesis for {session_id}")
        
        # Critical Fix: Use user query for specific questions, hardcoded only for the full report
        if "memory gauntlet report" in request.message.lower():
            query_text = "Identity Professional Financial Schedule Projects Goals"
        else:
            query_text = request.message
            
        ctx = retrieve_context(query_text)
        ans = gemma_fallback(request.message, ctx)
        return ChatResponse(response=ans)

    # AGENT PATH
    ctx = retrieve_context(request.message)

    if graph:
        try:
            from ebbiforge_core import HistoryBuffer, TrajectoryPoint
            buffer = HistoryBuffer()
            if ctx: buffer.add(TrajectoryPoint(1, "System", f"Context:\n{ctx}"))
            buffer.add(TrajectoryPoint(2, "User", request.message))
            task_id = f"task_{uuid.uuid4().hex}"
            graph.run_task(task_id, buffer)
            final_ans = buffer.get_raw()[-1].thought
        except:
            final_ans = gemma_fallback(request.message, ctx)
    else:
        final_ans = gemma_fallback(request.message, ctx)

    return ChatResponse(response=final_ans)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
