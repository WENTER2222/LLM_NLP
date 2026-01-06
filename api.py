from fastapi import FastAPI
from rag_chat import chat

app = FastAPI()

@app.get("/chat")
def chat_api(q: str):
    return {"answer": chat(q)}
