import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 使用中文效果很好的开源 embedding
model = SentenceTransformer("BAAI/bge-base-zh")

def embed(texts):
    return model.encode(texts, normalize_embeddings=True)

with open("data/law_docs.txt", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

embs = embed(texts)

dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs.astype("float32"))

faiss.write_index(index, "law.index")
np.save("law_texts.npy", texts)

print("向量库构建完成！")
