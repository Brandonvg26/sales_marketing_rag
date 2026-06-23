import os, numpy as np

def get_embedder():
    backend = os.environ["EMBED_BACKEND"]
    if backend == "openai":
        from openai import OpenAI
        client, model = OpenAI(), os.environ["EMBED_MODEL"]
        def embed(texts: list[str]) -> list[list[float]]:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        return embed
    elif backend == "local":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        def embed(texts: list[str]) -> list[list[float]]:
            return model.encode(texts, normalize_embeddings=True).tolist()
        return embed
    elif backend == "ollama":
        import requests
        model = os.environ["EMBED_MODEL"]
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/embeddings")
        def embed(texts: list[str]) -> list[list[float]]:
            return [requests.post(url, json={"model": model, "prompt": t}).json()["embedding"]
                    for t in texts]
        return embed
    raise ValueError(backend)


import os, numpy as np, psycopg
from pgvector.psycopg import register_vector
 
def _conn():
    c = psycopg.connect(os.environ["DATABASE_URL"]); register_vector(c); return c
 
def vector_search(conn, embed, query, k=20):
    qv = np.array(embed([query])[0])
    return conn.execute(
        "SELECT id, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (qv, k)).fetchall()
 
def keyword_search(conn, query, k=20):
    return conn.execute(
        """SELECT id, content
           FROM chunks
           WHERE tsv @@ plainto_tsquery('english', %s)
           ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC
           LIMIT %s""",
        (query, query, k)).fetchall()
 
def rrf_fuse(vec_rows, kw_rows, k=60, top_n=6):
    scores, content = {}, {}
    for rank, (cid, c) in enumerate(vec_rows):
        scores[cid] = scores.get(cid, 0) + 1/(k + rank); content[cid] = c
    for rank, (cid, c) in enumerate(kw_rows):
        scores[cid] = scores.get(cid, 0) + 1/(k + rank); content[cid] = c
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [{"id": cid, "content": content[cid]} for cid in ranked]
 
def retrieve(query, top_n=6):
    conn, embed = _conn(), get_embedder()
    v = vector_search(conn, embed, query)
    kw = keyword_search(conn, query)
    return rrf_fuse(v, kw, top_n=top_n)


SYSTEM = (
 "You are a sales & marketing intelligence assistant. Answer ONLY from the "
 "provided context. If the answer isn't in the context, say you don't know. "
 "Cite sources as [chunk_id] after each claim.")
 
def _format_context(chunks):
    return "\n\n".join(f"[{c['id']}] {c['content']}" for c in chunks)
 
def generate(query, chunks):
    ctx = _format_context(chunks)
    user = f"Context:\n{ctx}\n\nQuestion: {query}"
    backend = os.environ["GEN_BACKEND"]
    if backend == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=os.environ["CHAT_MODEL"], temperature=0,
            messages=[{"role":"system","content":SYSTEM},
                      {"role":"user","content":user}])
        return r.choices[0].message.content
    # if backend == "anthropic":
    #     import anthropic
    #     r = anthropic.Anthropic().messages.create(
    #         model=os.environ["CHAT_MODEL"], max_tokens=1024, system=SYSTEM,
    #         messages=[{"role":"user","content":user}])
    #     return r.content[0].text
    if backend == "ollama":
        import ollama
        r = ollama.chat(model=os.environ["CHAT_MODEL"],
            messages=[{"role":"system","content":SYSTEM},
                      {"role":"user","content":user}])
        return r["message"]["content"]
 
def answer(query, top_n=6):
    chunks = retrieve(query, top_n=top_n)
    return {"answer": generate(query, chunks),
            "contexts": [c["content"] for c in chunks],
            "chunk_ids": [c["id"] for c in chunks]}
