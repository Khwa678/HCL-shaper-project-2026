import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.chatbot_engine import SupportChatbotOrchestrator

app = FastAPI(
    title="LLM Customer Support Chatbot API",
    description="Context-Aware Support Chatbot with Memory, Intent Routing, and RAG Knowledge Retrieval",
    version="1.0.0"
)

orchestrator = SupportChatbotOrchestrator()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    turn_id: int
    user_query: str
    bot_response: str
    intent: str
    confidence_score: float
    escalated: int
    escalation_reason: str
    response_time_ms: int
    retrieved_context: List[Dict[str, Any]]

class CSATRequest(BaseModel):
    session_id: str
    satisfaction_score: int

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "LLM Customer Support Chatbot API",
        "ethical_disclosure": "This conversational service is AI-powered (Nova). All interactions are monitored for quality.",
        "endpoints": ["/api/chat", "/api/sessions/{session_id}", "/api/csat", "/api/faqs", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.session_id.strip() or not request.message.strip():
        raise HTTPException(status_code=400, detail="session_id and message are required.")
    
    result = orchestrator.process_message(
        session_id=request.session_id,
        user_query=request.message
    )
    return result

@app.get("/api/sessions/{session_id}")
def get_session_history(session_id: str):
    history = orchestrator.memory.get_session_history(session_id)
    return {
        "session_id": session_id,
        "total_turns": len(history),
        "is_escalated": orchestrator.memory.is_session_escalated(session_id),
        "history": history
    }

@app.post("/api/csat")
def submit_csat(request: CSATRequest):
    if request.satisfaction_score < 1 or request.satisfaction_score > 5:
        raise HTTPException(status_code=400, detail="satisfaction_score must be between 1 and 5.")
    orchestrator.memory.set_satisfaction_score(request.session_id, request.satisfaction_score)
    return {
        "status": "success",
        "session_id": request.session_id,
        "satisfaction_score": request.satisfaction_score
    }

@app.get("/api/faqs")
def list_faqs(category: Optional[str] = None):
    docs = orchestrator.rag_engine.documents
    if category:
        docs = [d for d in docs if d["category"] == category.lower()]
    return {"total": len(docs), "faqs": docs}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
