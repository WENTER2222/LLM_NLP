import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from config import DEEPSEEK_API_KEY

# embedding 模型
embed_model = SentenceTransformer("BAAI/bge-base-zh")

# DeepSeek LLM
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# 加载向量库
index = faiss.read_index("law.index")
texts = np.load("law_texts.npy", allow_pickle=True)

history = ""

def embed(texts):
    return embed_model.encode(texts, normalize_embeddings=True)

def retrieve(query, k=5):
    q_emb = embed([query])
    _, ids = index.search(q_emb.astype("float32"), k)
    return [texts[i] for i in ids[0]]

def chat(query):
    global history

    docs = retrieve(query)
    refs = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(docs)])

    prompt = f"""
你是一个中文法律专家。
规则：
1. 只能依据【参考资料】回答
2. 资料不足请回答：无法根据现有资料确定
3. 标注引用编号

【历史对话】
{history}

【参考资料】
{refs}

【问题】
{query}
"""

    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = r.choices[0].message.content
    history += f"\n用户：{query}\n助手：{answer}\n"
    return answer
