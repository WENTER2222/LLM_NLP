import gradio as gr
from rag_chat import chat

gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, label="法律问题"),
    outputs=gr.Textbox(lines=10, label="回答"),
    title="中文法律 RAG 问答系统（DeepSeek）"
).launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False
)


