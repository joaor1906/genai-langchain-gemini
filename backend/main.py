import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from models import ChatRequest, ChatResponse
from memory import build_memory

load_dotenv()

app = FastAPI(title="GenAI Chat Backend (Gemini)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração do Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY não definido no .env")

genai.configure(api_key=API_KEY)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        memory = build_memory(req.session_id)
        # Recupera histórico de mensagens (LangChain + SQLite)
        previous = memory.load_memory_variables({}).get("history", [])

        # Concatena histórico num texto simples (user/assistant)
        history_text = ""
        for msg in previous:
            role = "User" if msg.type == "human" else "Assistant"
            history_text += f"{role}: {msg.content}\n"

        prompt = f"""{history_text}User: {req.message}
Assistant:"""

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, generation_config={"temperature": TEMPERATURE})
        text = response.text or "(sem resposta)"

        # Guarda no histórico
        memory.save_context({"input": req.message}, {"output": text})

        return ChatResponse(session_id=req.session_id, answer=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
