import sys
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# Setup paths & logging
BASE_DIR = Path(__file__).parent.absolute()
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import traceback
from pydantic import BaseModel
import uuid
from datetime import datetime
import json
import asyncio
from typing import Optional, List
from openai import OpenAI
from sqlalchemy.orm import Session
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("WARNING: google-genai not found, fallback to legacy might be needed.")

# Import our modules
try:
    from database import engine, Base, get_db, SessionLocal
    from models import User, Conversation, Document, Log, Memory
    from auth_utils import verify_password, get_password_hash, create_access_token, decode_access_token
    from model_router import ModelRouter, ChatMode
    from cache_manager import ResponseCache
except ImportError as e:
    print(f"CRITICAL: Failed to import core modules: {e}")
    sys.exit(1)

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer(auto_error=False)

async def get_current_user(db: Session = Depends(get_db), auth: HTTPAuthorizationCredentials = Depends(security)):
    payload = decode_access_token(auth.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Handle RAG dependencies gracefully
try:
    from document_processor import DocumentProcessor
    HAS_RAG = True
except ImportError:
    try:
        from backend.document_processor import DocumentProcessor
        HAS_RAG = True
    except ImportError:
        logger.warning("⚠️ RAG dependencies (chromadb, etc.) not found. Running in standard mode.")
        HAS_RAG = False

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(title="EduBot Advanced", version="6.0.0")

# Configure paths
BASE_DIR = Path(__file__).parent.absolute()
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"

# Create directories
STATIC_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Serve static files
@app.get("/login")
async def login_page():
    return FileResponse("static/login.html")

# Health check endpoint for deployment platforms
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "6.0.0"}

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_trace = traceback.format_exc()
    logger.error(f"Global Error Catch: {str(exc)}\n{error_trace}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc), "trace": error_trace}
    )

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-1.5-pro")
SITE_URL = os.getenv("SITE_URL", "http://localhost:8000")
SITE_NAME = os.getenv("SITE_NAME", "EduBot Pro")

# Initialize Gemini
if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not found in environment")

# Initialize Document Processor (RAG) if available
if HAS_RAG:
    doc_processor = DocumentProcessor()
else:
    doc_processor = None

# Initialize Model Router and Cache
model_router = ModelRouter()
response_cache = ResponseCache(db_path=str(BASE_DIR / "cache.db"))

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    image_data: Optional[str] = None # Base64 image
    conversation_id: Optional[str] = None
    use_history: Optional[bool] = True
    temperature: Optional[float] = 0.5 # Lowered default for more focused "high results"
    token: Optional[str] = None # For streaming auth
    mode: Optional[str] = "default" # Education mode: exam, explain, beginner, advanced
    use_cache: Optional[bool] = True # Enable/disable cache
    use_premium: Optional[bool] = False # Enable LLaMA-4 models (paid tier)
    force_model: Optional[str] = None # 'gemini-pro' or 'gemini-flash'

class UserRegister(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_name: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: datetime

class QuestionRequest(BaseModel):
    question: str
    max_length: Optional[int] = 200

# Gemini Client
# Gemini Client (Modern SDK)
class GeminiClient:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.working_model = None # Remember successful model
        if self.api_key:
            try:
                # Use the new google-genai Client
                self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
                logger.info("✅ Gemini API (google-genai) initialized")
            except Exception as e:
                logger.error(f"Failed to initialize google-genai: {e}")
                self.client = None
        else:
            logger.warning("⚠️ GEMINI_API_KEY not found")
            self.client = None
    
    def _determine_model_complexity(self, message: str, has_image: bool) -> List[str]:
        # If we found a working model, try it first!
        models = []
        if self.working_model:
            models.append(self.working_model)
            
        # Priority list (excluding 1.5 which fails)
        base_list = [
            "gemini-2.0-flash-001",
            "gemini-2.0-flash", 
            "gemini-2.0-flash-exp", 
            "gemini-2.5-flash"
        ]
        
        for m in base_list:
            if m != self.working_model:
                models.append(m)
                
        return models

    async def generate_response(self, messages: List[dict], temperature: float = 0.7, has_image: bool = False) -> str:
        if not self.client: return "Gemini API not configured."
        
        models_to_try = self._determine_model_complexity("", has_image)
        contents = self._convert_messages(messages)

        errors = []
        for model_name in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=8192,
                    )
                )
                self.working_model = model_name # Save success!
                return response.text
            except Exception as e:
                logger.warning(f"Failed with {model_name}: {e}")
                errors.append(f"{model_name}: {str(e)}")
                continue
        
        return f"All Gemini models failed. Errors: {'; '.join(errors)}"

    async def generate_streaming_response(self, messages: List[dict], temperature: float = 0.7, has_image: bool = False):
        if not self.client:
            yield "Gemini API not configured."
            return

        models_to_try = self._determine_model_complexity("", has_image)
        contents = self._convert_messages(messages)

        for model_name in models_to_try:
            try:
                stream = self.client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=8192,
                    )
                )
                for chunk in stream:
                    if chunk.text:
                        yield chunk.text
                return 
            except Exception as e:
                logger.warning(f"Streaming failed with {model_name}: {e}")
                continue
        yield "⚠️ All Gemini models failed to respond."

    def _convert_messages(self, messages: List[dict]):
        contents = []
        system_prompt = ""
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
                continue
            
            role = "user" if m["role"] == "user" else "model"
            parts = []
            
            content = m["content"]
            if isinstance(content, list):
                for p in content:
                    if p["type"] == "text":
                        parts.append(types.Part(text=p["text"]))
                    elif p["type"] == "image_url":
                        b64_data = p["image_url"]["url"].split(",")[1]
                        parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=b64_data)))
            else:
                parts.append(types.Part(text=content))
                
            contents.append(types.Content(role=role, parts=parts))
        
        # Prepend system prompt to the first user message if present
        if system_prompt and contents:
            for c in contents:
                if c.role == "user":
                    c.parts[0].text = f"{system_prompt}\n\n{c.parts[0].text}"
                    break
        return contents

# OpenRouter Client (Fallback)
class OpenRouterClient:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL
        self.model = DEFAULT_MODEL
        
        if not self.api_key:
            logger.warning("⚠️ OPENROUTER_API_KEY not found")
            self.enabled = False
            self.client = None
        else:
            self.enabled = True
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            logger.info(f"✅ OpenRouter API initialized (fallback)")
    
    async def generate_response(self, messages: List[dict], temperature: float = 0.7, model_name: Optional[str] = None) -> str:
        if not self.enabled:
            return "AI services are currently unavailable. Please configure API keys."
        
        # Use specific model if provided, otherwise try fallback list
        models_to_try = [model_name] if model_name else [
            "google/gemini-pro-1.5",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-flash-1.5:free",
            "deepseek/deepseek-chat:free"
        ]
        
        for model in models_to_try:
            try:
                completion = self.client.chat.completions.create(
                    extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_NAME},
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2000
                )
                if completion.choices:
                    logger.info(f"✅ OpenRouter response from {model}")
                    return completion.choices[0].message.content
            except Exception as e:
                logger.error(f"Model {model} failed: {e}")
                continue
        
        return "I'm having trouble connecting right now. Please try again."

gemini_client = GeminiClient()
openrouter_client = OpenRouterClient()

def log_activity(db: Session, user_id: str, activity: str, data: dict = None):
    try:
        new_log = Log(
            id=str(uuid.uuid4()),
            user_id=user_id,
            activity=activity,
            data=json.dumps(data) if data else None
        )
        db.add(new_log)
        db.commit()
    except Exception as e:
        logger.error(f"Activity logging failed: {e}")

async def generate_ai_enhanced_response(
    message: str, 
    db: Session, 
    conversation_history: List[dict] = None, 
    image_data: str = None,
    mode: ChatMode = ChatMode.DEFAULT,
    use_cache: bool = True,
    force_model: str = None
) -> str:
    """Enhanced response generation with model routing and caching."""
    
    # 1. Get RAG context
    rag_context = ""
    has_document = False
    if HAS_RAG and doc_processor:
        rag_context = doc_processor.get_relevant_context(message)
        has_document = bool(rag_context)
    
    # 2. Check cache (if enabled)
    if use_cache:
        cached = response_cache.get(message, model_name="auto", doc_ids=None)
        if cached:
            logger.info("📦 Returning cached response")
            return cached.get("response", "")
    
    # 3. Determine model via router
    routing_decision = model_router.determine_model(
        message=message,
        has_image=bool(image_data),
        has_document=has_document,
        mode=mode,
        use_premium=use_cache,  # Pass premium flag
        force_model=force_model # Fixed: Use the parameter from signature
    )
    
    model_type = routing_decision["model_type"]
    model_config = routing_decision["config"]
    logger.info(f"🧠 Model Router: {model_type if isinstance(model_type, str) else model_type.value} - {routing_decision['reasoning']}")
    
    # 4. Prepare system prompt (mode-aware)
    system_prompt = model_router.get_system_prompt(mode, rag_context)
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 5. Add history
    if conversation_history:
        for msg in conversation_history[-12:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # 6. Add current message (multimodal)
    if image_data:
        user_content = [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": message})
    
    # 7. Generate response based on provider
    response = ""
    temperature = model_router.get_temperature(mode, model_type)
    
    if model_config["provider"] == "gemini":
        response = await gemini_client.generate_response(messages, temperature=temperature, has_image=bool(image_data))
    elif model_config["provider"] == "openrouter":
        response = await openrouter_client.generate_response(messages, temperature=temperature, model_name=model_config["name"])
    
    if not response:
        logger.warning("Primary model failed, using fallback")
        response = await gemini_client.generate_response(messages, temperature=temperature, has_image=bool(image_data))
    
    # 8. Cache response
    if use_cache and response:
        response_cache.set(message, "auto", {"response": response}, doc_ids=None, ttl_hours=24)
    
    return response

@app.get("/")
async def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest, 
    db: Session = Depends(get_db),
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Streaming endpoint for real-time responses with robust auth."""
    from fastapi.responses import StreamingResponse
    import json
    
    async def event_generator():
        # Use a fresh session for the generator to ensure persistence across long streams
        db_gen = SessionLocal()
        try:
            user_id = None
            if auth and auth.credentials:
                payload = decode_access_token(auth.credentials)
                if payload: 
                    user_id = payload.get("sub")
                    logger.info(f"🔑 Auth from Header: {user_id}")
            
            if not user_id and request.token:
                payload = decode_access_token(request.token)
                if payload: 
                    user_id = payload.get("sub")
                    logger.info(f"🔑 Auth from JSON: {user_id}")
            
            if not user_id:
                logger.warning("🚫 No User ID found in request")
            
            logger.info(f"📁 Stream Chat: User={user_id}, conv_id={request.conversation_id}")

            conversation_id = request.conversation_id
            conv = None
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                conv = Conversation(
                    id=conversation_id, 
                    title=request.message[:40] + "...",
                    user_id=user_id,
                    chat_history="[]"
                )
                db_gen.add(conv)
                db_gen.commit()
            else:
                conv = db_gen.query(Conversation).filter(Conversation.id == conversation_id).first()
                if conv and conv.user_id and conv.user_id != user_id:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Unauthorized conversation access'})}\n\n"
                    return
            
            # 3. Prepare messages & load history
            history = json.loads(conv.chat_history) if (conv and conv.chat_history) else []
            
            rag_context = ""
            if HAS_RAG and doc_processor:
                rag_context = doc_processor.get_relevant_context(request.message)
            
            system_prompt = f"""You are Mega Ai, an elite AI educational assistant.
Your mission is to provide exceptionally accurate, comprehensive, and insightful responses.

CORE CAPABILITIES:
- **Deep Reasoning**: Break down complex topics with step-by-step logical analysis
- **Visual Intelligence**: Analyze images with meticulous detail and context
- **Document Mastery**: Synthesize information from uploaded documents
- **Conversational Memory**: Leverage full conversation history for coherent dialogue
- **Multimodal Excellence**: Handle text, images, and documents seamlessly

RESPONSE STANDARDS:
- Accuracy is paramount - provide factual, well-reasoned answers
- Use clear structure with headers, lists, tables, and code blocks (Markdown)
- For math/science: show work, equations, and visual explanations
- For images: describe composition, objects, text, context, and implications
- For coding: provide working examples with explanations
- Cite document context when relevant

RELEVANT DOCUMENT CONTEXT:
\"\"\"
{rag_context if rag_context else "No additional document context available."}
\"\"\""""
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add limited history
            if history:
                for msg in history[-12:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Current message
            if request.image_data:
                user_content = [
                    {"type": "text", "text": request.message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_data}"}}
                ]
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": request.message})
            
            # Determine model via router
            routing_decision = model_router.determine_model(
                message=request.message,
                has_image=bool(request.image_data),
                has_document=bool(rag_context),
                mode=ChatMode.DEFAULT, # Keep default for now or parse from request if added
                use_premium=True,
                force_model=request.force_model
            )
            model_config = routing_decision["config"]
            logger.info(f"🧠 Stream Model: {routing_decision['model_type']} - {routing_decision['reasoning']}")

            # 4. Stream response
            # Send conversation ID first
            yield f"data: {json.dumps({'type': 'conversation_id', 'id': conversation_id})}\n\n"
            
            full_response = ""
            # Always try Gemini first for streaming (most reliable)
            try:
                async for chunk in gemini_client.generate_streaming_response(messages, temperature=request.temperature, has_image=bool(request.image_data)):
                    if chunk and not chunk.startswith("⚠️"):
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'content', 'text': chunk})}\n\n"
                    elif chunk.startswith("⚠️"):
                        # Gemini failed, try OpenRouter fallback
                        logger.warning("Gemini streaming failed, trying OpenRouter fallback")
                        response = await openrouter_client.generate_response(messages, temperature=request.temperature)
                        if response and "trouble connecting" not in response:
                            full_response = response
                            yield f"data: {json.dumps({'type': 'content', 'text': response})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'error', 'message': 'AI services temporarily unavailable. Please check API configuration.'})}\n\n"
                            return
            except Exception as stream_error:
                logger.error(f"Streaming error: {stream_error}")
                # Fallback to non-streaming
                try:
                    response = await openrouter_client.generate_response(messages, temperature=request.temperature)
                    if response and "trouble connecting" not in response:
                        full_response = response
                        yield f"data: {json.dumps({'type': 'content', 'text': response})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'AI services temporarily unavailable. Please try again.'})}\n\n"
                        return
                except:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to connect to AI services.'})}\n\n"
                    return
            
            # 5. Save history back to JSON
            history.append({
                "role": "user", 
                "content": request.message, 
                "image_data": request.image_data,
                "timestamp": datetime.now().isoformat()
            })
            history.append({
                "role": "assistant", 
                "content": full_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Re-fetch conversation to ensure session is healthy
            conv = db_gen.query(Conversation).filter(Conversation.id == conversation_id).first()
            if conv:
                conv.chat_history = json.dumps(history)
                db_gen.commit()
                # Log Activity
                log_activity(db_gen, user_id, "chat", {"conversation_id": conversation_id, "message_length": len(request.message)})
            
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            db_gen.close()
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Non-streaming chat endpoint (updated to use JSON history)
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    db: Session = Depends(get_db),
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    try:
        user_id = None
        if auth and auth.credentials:
            payload = decode_access_token(auth.credentials)
            if payload: user_id = payload.get("sub")
        
        if not user_id and request.token:
            payload = decode_access_token(request.token)
            if payload: user_id = payload.get("sub")
        
        logger.info(f"📁 Sync Chat: User={user_id}, conv_id={request.conversation_id}")

        # 1. Get or create conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            new_conv = Conversation(id=conversation_id, title=request.message[:40] + "...", user_id=user_id)
            db.add(new_conv)
            db.commit()
            conv = new_conv
        else:
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if conv and conv.user_id and conv.user_id != user_id:
                raise HTTPException(status_code=403, detail="Unauthorized")
        
        # 2. Load history from JSON
        history = json.loads(conv.chat_history) if conv.chat_history else []
        
        # 3. Parse mode
        try:
            chat_mode = ChatMode(request.mode)
        except ValueError:
            chat_mode = ChatMode.DEFAULT
        
        # 4. Generate response with routing
        response_text = await generate_ai_enhanced_response(
            message=request.message,
            db=db,
            conversation_history=history,
            image_data=request.image_data,
            mode=chat_mode,
            use_cache=request.use_cache,
            force_model=request.force_model # Pass through
        )
        
        # 5. Update history
        history.append({"role": "user", "content": request.message, "image_data": request.image_data, "timestamp": datetime.now().isoformat()})
        history.append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        conv.chat_history = json.dumps(history)
        db.commit()
        
        log_activity(db, user_id, "chat_sync", {"conversation_id": conversation_id, "mode": request.mode})
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            message_id=str(uuid.uuid4()), # Placeholder for compat
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not HAS_RAG:
        raise HTTPException(status_code=500, detail="Document analysis (RAG) is currently unavailable due to missing dependencies.")
    
    try:
        file_id = str(uuid.uuid4())
        file_path = UPLOADS_DIR / f"{file_id}_{file.filename}"
        
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Process for RAG
        extracted_text = await doc_processor.add_document(str(file_path), file.filename, file_id)
        
        # Store in DB
        new_doc = Document(
            id=file_id, 
            filename=file.filename, 
            file_path=str(file_path),
            extracted_text=extracted_text[:2000] if extracted_text else None
        )
        db.add(new_doc)
        db.commit()
        
        return {"status": "success", "file_id": file_id, "filename": file.filename}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Authentication Endpoints ---

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(data: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    new_user = User(
        id=user_id,
        email=data.email,
        hashed_password=get_password_hash(data.password),
        name=data.name
    )
    db.add(new_user)
    db.commit()
    
    access_token = create_access_token(data={"sub": user_id})
    log_activity(db, user_id, "register", {"email": data.email})
    return {"access_token": access_token, "token_type": "bearer", "user_name": data.name}

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not user.hashed_password or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user.id})
    log_activity(db, user.id, "login")
    return {"access_token": access_token, "token_type": "bearer", "user_name": user.name}

# --- Data Persistence & History ---

@app.get("/api/conversations")
async def list_conversations(db: Session = Depends(get_db), auth: HTTPAuthorizationCredentials = Depends(security)):
    user_payload = decode_access_token(auth.credentials)
    if not user_payload: return {"conversations": []}
    
    user_id = user_payload.get("sub")
    convs = db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.created_at.desc()).all()
    return {"conversations": [{"id": c.id, "title": c.title, "date": c.created_at} for c in convs]}

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, db: Session = Depends(get_db), auth: HTTPAuthorizationCredentials = Depends(security)):
    user_payload = decode_access_token(auth.credentials)
    if not user_payload: raise HTTPException(status_code=401)
    
    user_id = user_payload.get("sub")
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
        
    history = json.loads(conv.chat_history) if conv.chat_history else []
    return {
        "conversation_id": conversation_id,
        "messages": history
    }

@app.delete("/api/conversations/{conversation_id}")
async def delete_history_item(conversation_id: str, db: Session = Depends(get_db), auth: HTTPAuthorizationCredentials = Depends(security)):
    user_payload = decode_access_token(auth.credentials)
    if not user_payload: raise HTTPException(status_code=401)
    
    user_id = user_payload.get("sub")
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id).first()
    if conv:
        db.delete(conv)
        db.commit()
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Conversation not found or access denied")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "api": "Mega Ai v7.0 - Smart Routing Enabled", 
        "features": {
            "model_routing": True,
            "response_caching": True,
            "rag_enabled": HAS_RAG,
            "education_modes": ["exam", "explain", "beginner", "advanced", "default"]
        }
    }

@app.get("/api/cache/stats")
async def cache_stats(auth: HTTPAuthorizationCredentials = Depends(security)):
    """Get cache statistics (admin only)."""
    # For now, allow any authenticated user to see stats
    user_payload = decode_access_token(auth.credentials)
    if not user_payload:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    stats = response_cache.stats()
    return stats

@app.post("/api/cache/clear")
async def clear_cache(auth: HTTPAuthorizationCredentials = Depends(security)):
    """Clear the response cache (admin only)."""
    user_payload = decode_access_token(auth.credentials)
    if not user_payload:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    success = response_cache.clear_all()
    return {"status": "success" if success else "failed", "message": "Cache cleared"}

@app.post("/api/cache/cleanup")
async def cleanup_cache():
    """Cleanup expired cache entries - can be called by cron."""
    deleted = response_cache.cleanup_expired()
    return {"status": "success", "deleted_entries": deleted}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("openrouter_main:app", host="127.0.0.1", port=8000, reload=True)
