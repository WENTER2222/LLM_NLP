from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from rag_chat import chat

app = FastAPI()

# 保留多轮对话历史
history_html = ""

@app.get("/", response_class=HTMLResponse)
def home():
    global history_html
    return f"""
    <html>
        <head>
            <title>中文法律 RAG 问答系统</title>
        </head>
        <body>
            <h2>中文法律 RAG 问答系统</h2>
            <form method="post" action="/chat">
                <input name="question" type="text" placeholder="请输入问题" style="width:400px;">
                <input type="submit" value="提交">
            </form>
            <hr>
            <div style="white-space:pre-wrap;">{history_html}</div>
        </body>
    </html>
    """

@app.post("/chat", response_class=HTMLResponse)
def chat_post(question: str = Form(...)):
    global history_html
    answer = chat(question)
    # 保存历史对话到 HTML 中
    history_html += f"用户：{question}\n助手：{answer}\n\n"
    return home()

# 可选：运行 uvicorn 时直接访问 http://127.0.0.1:8000/
# uvicorn api:app --host 127.0.0.1 --port 8000
