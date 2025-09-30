import os
import re
import json
import argparse
from typing import Iterable, List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
from contextlib import asynccontextmanager

#  Config 
load_dotenv()

DATA_FILE      = os.getenv("DATA_FILE", "FRW-J.json")
INDEX_DIR      = os.getenv("INDEX_DIR", "./chroma_frw")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE   = os.getenv("EMBED_DEVICE", "cpu")  

OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
LLM_MODEL      = os.getenv("LLM_MODEL", "mistral")
MAX_TOKENS_OUT = int(os.getenv("MAX_TOKENS", "448"))  
KEEP_ALIVE     = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
OLLAMA_SEED    = int(os.getenv("OLLAMA_SEED", "42"))  

CHUNK_TOKENS   = int(os.getenv("CHUNK_TOKENS", "700"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K_DEFAULT  = int(os.getenv("TOP_K", "3"))  

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "frw_lore")

# PERSONA + EXPRESSÕES
EXPRESSIONS = [
    "Neutral", "Confident", "Surprised", "Annoyed", "Thoughtful", "Sad"
]

# Saída **determinística e estável**: pedimos JSON e depois renderizamos 
SYSTEM_PROMPT = f"""
ROLE: You are an elder librarian NPC from the Forgotten Realms, kind and helpful.
VOICE: First-person, kindly mentor; address the reader as "adventurer" at least once.
STYLE: Warm, patient, slightly poetic but objective; 1 light metaphor max.
PARAPHRASE: Rephrase in your own words; never copy ≥5 consecutive words from CONTEXT (proper names are ok).
SCOPE: Answer ONLY using the passages in CONTEXT. If info is missing, say you don't know and invite the adventurer to bring more scrolls.

SENSITIVE CONTENT POLICY:
- Do NOT use or repeat explicit terms of sexual violence, slurs, or graphic gore.
- If the CONTEXT mentions such topics, refer to them obliquely (e.g., "a violent origin myth", "assault", "harm")—never the explicit term itself.

OUTPUT FORMAT (REQUIRED): Return ONLY a valid JSON object matching this schema (no markdown, no extra text):
{{
  "paragraphs": [
    {{"text": "<string with <= 2 sentences and <= 60 words>", "tag": "<one of {', '.join(EXPRESSIONS)}>"}},
    ... (2 to 3 items total)
  ]
}}

RULES:
- Produce 2 to 3 items in "paragraphs".
- Each "text" must be plain prose (no angle brackets, no brackets) and must NOT include the tag.
- Each "tag" must be exactly one of: {', '.join(EXPRESSIONS)}.
- Do NOT include any fields other than "paragraphs".
- Return ONLY JSON.
"""

# Utils 

def iter_dataset_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print("Erro ao decodificar linha:", e)
                continue

def chunk_text(txt: str, max_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> Iterable[str]:
    words = txt.split()
    if not words:
        return
    step = max_tokens - overlap
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + max_tokens])
        if len(piece.split()) > 30:
            yield piece

def make_embedder():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device=EMBED_DEVICE,
    )

def make_collection():
    client = chromadb.PersistentClient(path=INDEX_DIR)
    embedder = make_embedder()
    return client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=embedder,
    )

# Build Index

def cmd_build_index(data_file: str):
    col = make_collection()
    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_metas: List[dict] = []
    batch_size = int(os.getenv("INDEX_BATCH_SIZE", "500"))

    with tqdm(desc="Indexando páginas", unit="pag") as pbar:
        for rec in iter_dataset_json(data_file):
            title = rec.get("page", "Untitled")
            text = rec.get("content", "")
            if not text:
                continue
            for i, ck in enumerate(chunk_text(text)):
                batch_ids.append(f"{title}::{i}")
                batch_docs.append(ck)
                batch_metas.append({"title": title, "chunk_i": i})

                if len(batch_ids) >= batch_size:
                    col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                    batch_ids, batch_docs, batch_metas = [], [], []
            pbar.update(1)

    if batch_ids:
        col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

# Rendering 

# Escolha de pontuação por tag (exclamação p/ emoção)
TERMINAL_PUNCT_BY_TAG = {
    "Surprised": "!",
    "Annoyed": "!",
    "Confident": ".",
    "Thoughtful": ".",
    "Sad": ".",
    "Neutral": ".",
}

# Escolha de pontuação por tag
TERMINAL_PUNCT_BY_TAG = {
    "Surprised": "!",
    "Annoyed": "!",
    "Confident": ".",
    "Thoughtful": ".",
    "Sad": ".",
    "Neutral": ".",
}

def ensure_terminal_punct(txt: str, tag: str) -> str:
    
    # Garante pontuação final antes da tag - se houver aspas/fecha-parênteses no fim, insere a pontuação antes deles
    
    txt = txt.rstrip()
    if not txt:
        return txt
    # separa sufixos - aspas/parênteses
    m = re.match(r"^(.*?)([\"'’”)\]]+)?\s*$", txt)
    base = m.group(1) if m else txt
    trail = m.group(2) if (m and m.group(2)) else ""
    # verifica se em pontuacao
    if re.search(r"[.!?…]$", base):
        return base + trail
    p = TERMINAL_PUNCT_BY_TAG.get(tag, ".")
    return base + p + trail


def render_answer_from_json(data: dict, allowed_tags: List[str]) -> str:
    
    #Converte o JSON em texto final e retorna string com 2–3 parágrafos, cada um terminando em [Tag]
    
    paras = data.get("paragraphs")
    if not isinstance(paras, list):
        raise ValueError("JSON sem 'paragraphs' lista")

    out_parts: List[str] = []
    allowed = set(allowed_tags)

    for item in paras[:3]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        tag  = str(item.get("tag", "")).strip()
        if not text:
            continue
        if tag not in allowed:
            tag = "Neutral"

        text = ensure_terminal_punct(text, tag)
        out_parts.append(f"{text} [{tag}]")

    # garantir pelo menos 2 paragarfos
    if len(out_parts) == 1:
        out_parts.append("[Neutral]")

    return "\n\n".join(out_parts[:3])

def extract_first_json_block(s: str) -> str | None:
    # Extrai o primeiro objeto/array JSON bem-formado dentro de uma string com ruído
    s = s.strip()

    # Se vier em code fence ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass

    # Tenta direto
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    # Varre do primeiro { ou [
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    stack = []
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if not stack:
                    return None
                opening = stack.pop()
                if (opening == "{" and c != "}") or (opening == "[" and c != "]"):
                    return None
                if not stack:
                    return s[start : i + 1]

    # fallback: corta no último } ou ]
    end = max(s.rfind("}"), s.rfind("]"))
    if end > (start or -1):
        frag = s[start : end + 1]
        try:
            json.loads(frag)
            return frag
        except Exception:
            return None
    return None


def parse_llm_json(raw) -> dict | list:
    # Aceita dict/list já prontos ou string; tenta extrair/parsear JSON
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        raw = str(raw)
    block = extract_first_json_block(raw)
    if block is None:
        raise ValueError("Nenhum JSON encontrado na resposta do LLM.")
    return json.loads(block)


def normalize_to_paragraphs(obj) -> dict:
    
    # Converte variações comuns para o esquema canônico - {"paragraphs":[{"text":..., "tag":...}, ...]}
    
    # Já no formato
    if isinstance(obj, dict) and isinstance(obj.get("paragraphs"), list):
        return obj

    # {"answer": [...]}
    if isinstance(obj, dict) and isinstance(obj.get("answer"), list):
        paras = []
        for s in obj["answer"][:3]:
            t = str(s).strip()
            m = re.search(r"\[\s*(\w+)\s*\]\s*$", t)
            tag = "Neutral"
            if m and m.group(1) in EXPRESSIONS:
                tag = m.group(1)
                t = t[: m.start()].rstrip()
            paras.append({"text": t, "tag": tag})
        return {"paragraphs": paras}

    # Lista simples -> cada item vira parágrafo neutro
    if isinstance(obj, list):
        paras = [{"text": str(x).strip(), "tag": "Neutral"} for x in obj[:3]]
        return {"paragraphs": paras}

    # string simples -> quebra em 2–3 blocos
    if isinstance(obj, str):
        parts = [p.strip() for p in re.split(r"\r?\n\s*\r?\n", obj) if p.strip()]
        if len(parts) < 2:
            sents = re.split(r"(?<=[.!?])\s+", obj)
            parts, bucket = [], []
            for s in sents:
                bucket.append(s)
                if len(" ".join(bucket)) > 160 and len(parts) < 2:
                    parts.append(" ".join(bucket))
                    bucket = []
            if bucket:
                parts.append(" ".join(bucket))
            parts = [p.strip() for p in parts if p.strip()] or [obj.strip()]
            if len(parts) == 1:
                parts.append("")
        paras = [{"text": p, "tag": "Neutral"} for p in parts[:3]]
        return {"paragraphs": paras}

    raise ValueError("Objeto não convertido para esquema 'paragraphs'.")


def repair_output(text: str, allowed_tags: List[str]) -> str:
    #Fallback quando JSON não é válido - tentar impor 2–3 parágrafos com tag final
    parts = [p.strip() for p in re.split(r"\r?\n\s*\r?\n", text or "") if p.strip()]
    if len(parts) < 2:
        # quebra por sentenças
        sents = re.split(r"(?<=[.!?])\s+", text or "")
        parts, bucket = [], []
        for s in sents:
            bucket.append(s)
            if len(" ".join(bucket)) > 160 and len(parts) < 2:
                parts.append(" ".join(bucket))
                bucket = []
        if bucket:
            parts.append(" ".join(bucket))
        parts = [p.strip() for p in parts if p.strip()] or [text.strip()]
        if len(parts) == 1:
            parts.append("")

    alt = "|".join(map(re.escape, allowed_tags))
    tag_end = re.compile(r"\[\s*(?:%s)\s*\]\s*\Z" % alt, re.IGNORECASE)

    cleaned: List[str] = []
    for p in parts[:3]:
        p = p.replace("\u200b", "")
        if tag_end.search(p):
            cleaned.append(p.strip())
        else:
            cleaned.append((p.rstrip(" .!?\t\r\n") + " [Neutral]").strip())

    while len(cleaned) < 2:
        cleaned.append("[Neutral]")

    return "\n\n".join(cleaned[:3])

# API

HttpClient: httpx.Client | None = None
Collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global HttpClient, Collection
    HttpClient = httpx.Client(
        timeout=httpx.Timeout(  
            connect=5.0,  
            read=300.0,  
            write=30.0,  
            pool=90.0  
        ),
        headers={"Connection": "keep-alive"}
    )

    Collection = make_collection()
    yield
    if HttpClient:
        HttpClient.close()

app = FastAPI(title="FRW RAG API (Optimized)", lifespan=lifespan)

class Ask(BaseModel):
    question: str
    k: int = TOP_K_DEFAULT


def build_prompt(question: str, contexts: List[Tuple[str, dict]]):
    ctx_block = "\n\n".join(
        f"{doc}\n(TITLE: {meta.get('title','')})" for doc, meta in contexts
    )
    return (
        f"CONTEXT:\n{ctx_block}\n\n"
        f"QUESTION: {question}\n"
        f"Follow the SYSTEM instructions strictly."
    )


@app.post("/ask")
def ask(body: Ask):
    res = Collection.query(
        query_texts=[body.question],
        n_results=body.k,
        include=["documents", "metadatas"],
    )
    contexts = list(zip(res["documents"][0], res["metadatas"][0]))
    prompt = build_prompt(body.question, contexts)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": prompt }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": OLLAMA_SEED,
            "num_predict": MAX_TOKENS_OUT
        },
    }

    r = HttpClient.post(OLLAMA_URL, json=payload)
    r.raise_for_status()

    resp_json = r.json()
    raw = resp_json.get("message", {}).get("content", "")

    try:
        obj = parse_llm_json(raw)
        obj = normalize_to_paragraphs(obj)
        out = render_answer_from_json(obj, EXPRESSIONS)
    except Exception:
        out = repair_output(str(raw), EXPRESSIONS)

    return {"reply": out}


@app.post("/ask_stream")
def ask_stream(body: Ask):
    res = Collection.query(
        query_texts=[body.question],
        n_results=body.k,
        include=["documents", "metadatas"],
    )
    contexts = list(zip(res["documents"][0], res["metadatas"][0]))
    prompt = build_prompt(body.question, contexts)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": prompt }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": OLLAMA_SEED,
            "num_predict": MAX_TOKENS_OUT
        },
    }

    buf: List[str] = []
    with HttpClient.stream("POST", OLLAMA_URL, json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            # ollama envia chunks de texto do campo 'response'
            if "response" in data:
                buf.append(data["response"]["content"])
    raw = "".join(buf)

    try:
        obj = parse_llm_json(raw)
        obj = normalize_to_paragraphs(obj)
        out = render_answer_from_json(obj, EXPRESSIONS)
    except Exception:
        out = repair_output(str(raw), EXPRESSIONS)

    return {"reply": out}


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build-index")
    p_build.add_argument("--data", default=DATA_FILE)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="Dev mode: reload on changes")

    args = parser.parse_args()
    if args.cmd == "build-index":
        cmd_build_index(args.data)
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run(
            "rag_frw:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
