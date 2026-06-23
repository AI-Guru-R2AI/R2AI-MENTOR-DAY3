import os
# Load dotenv
import sys
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)
dotenv_path = os.path.join(workspace_root, ".env")
load_dotenv(dotenv_path)

app = FastAPI(title="Apple Legal RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

def init_resources():
    """
    Khởi tạo các tài nguyên hệ thống (RAGPipeline) khi khởi động FastAPI server.
    """
    global pipeline
    try:
        from helpers.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        print("RAGPipeline loaded successfully in FastAPI server.")
    except Exception as e:
        print(f"Error initializing resources: {e}")
        traceback.print_exc()

# Initialize resources on startup
init_resources()

# Request model
class ChatRequest(BaseModel):
    """
    Pydantic model mô tả cấu trúc dữ liệu yêu cầu gửi lên từ client.
    """
    query: str
    reasoning: bool = True
    top_k_retrieval: int = 5
    top_k_rerank: int = 20
    temperature: float = 0.0
    top_k: int = 20
    top_p: float = 1.0
    min_p: float = 0.0
    enable_reranker: bool = False
    enable_prompt_guardrail: bool = False
    enable_stream_loop_guardrail: bool = False
    enable_grounding_guardrail: bool = False
    enable_language_guardrail: bool = False

@app.get("/api/status")
async def get_status():
    """
    Endpoint kiểm tra trạng thái hoạt động của hệ thống, bao gồm kết nối đến 
    Qdrant DB, các LLM model và thông tin cấu hình hiện tại.
    """
    qdrant_ok = False
    llm_ok = False
    
    if pipeline is not None and pipeline.db_client is not None:
        try:
            pipeline.db_client.client.get_collections()
            qdrant_ok = True
        except Exception:
            pass
            
    if pipeline is not None and pipeline.reasoning_llm_client is not None:
        try:
            pipeline.reasoning_llm_client.models.list()
            llm_ok = True
        except Exception:
            pass
            
    return {
        "status": "online" if (qdrant_ok and llm_ok) else "partial_or_offline",
        "qdrant": "connected" if qdrant_ok else "disconnected",
        "qdrant_details": f"{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')} / {pipeline.db_client.collection_name}" if pipeline else "N/A",
        "llm": "connected" if llm_ok else "disconnected",
        "llm_model": pipeline.reasoning_llm_client.default_model if pipeline else "N/A",
        "reasoning_model": pipeline.reasoning_llm_client.default_model if pipeline else "N/A",
        "non_reasoning_model": pipeline.non_reasoning_llm_client.default_model if pipeline else "N/A",
        "embedding_model": pipeline.embed_client.default_model if (pipeline and pipeline.embed_client) else "not_configured",
        "reranker_model": pipeline.rerank_client.default_model if (pipeline and pipeline.rerank_client) else "not_configured"
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Endpoint xử lý chat đồng bộ. Nhận câu hỏi, tìm kiếm tài liệu tương quan 
    và dùng LLM sinh câu trả lời có kiểm chứng.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    return pipeline.retrieve_and_answer(
        query=request.query,
        reasoning=request.reasoning,
        top_k_retrieval=request.top_k_retrieval,
        top_k_rerank=request.top_k_rerank,
        temperature=request.temperature,
        top_k_sampling=request.top_k,
        top_p=request.top_p,
        min_p=request.min_p,
        enable_reranker=request.enable_reranker,
        enable_prompt_guardrail=request.enable_prompt_guardrail,
        enable_stream_loop_guardrail=request.enable_stream_loop_guardrail,
        enable_grounding_guardrail=request.enable_grounding_guardrail,
        enable_language_guardrail=request.enable_language_guardrail
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Endpoint xử lý chat bất đồng bộ theo luồng dữ liệu (streaming) sử dụng Server-Sent Events (SSE).
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    generator = pipeline.retrieve_and_answer_stream(
        query=request.query,
        reasoning=request.reasoning,
        top_k_retrieval=request.top_k_retrieval,
        top_k_rerank=request.top_k_rerank,
        temperature=request.temperature,
        top_k_sampling=request.top_k,
        top_p=request.top_p,
        min_p=request.min_p,
        enable_reranker=request.enable_reranker,
        enable_prompt_guardrail=request.enable_prompt_guardrail,
        enable_stream_loop_guardrail=request.enable_stream_loop_guardrail,
        enable_grounding_guardrail=request.enable_grounding_guardrail,
        enable_language_guardrail=request.enable_language_guardrail
    )

    return StreamingResponse(generator, media_type='text/event-stream')

# Serve static frontend files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
async def get_index():
    """
    Endpoint trả về trang giao diện chính (index.html) của Apple Legal RAG Chatbot.
    """
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

