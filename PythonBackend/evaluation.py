import json
import re
import httpx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Config
API_URL = "http://localhost:8000/ask"
MODEL_NAME = "all-MiniLM-L6-v2"
DATASET_FILE = "C:\\FaculdadeDev\\ODS3Atividade1\\PythonBackend\\eval.jsonl"

# Utils 
model = SentenceTransformer(MODEL_NAME)
client = httpx.Client(timeout=30.0)  # aumentei timeout

def keyword_coverage(resposta, expected_keywords):
    resposta_lower = resposta.lower()
    return sum(1 for kw in expected_keywords if kw.lower() in resposta_lower) / len(expected_keywords)

def remove_tags(text):
    return re.sub(r"\[[^\]]+\]", "", text).strip()

# Avaliação
results = []

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

for i, entry in enumerate(entries):
    question = entry["question"]
    ground_truth = entry["ground_truth"]
    keywords = entry.get("expected_keywords", [])

    try:
        response = client.post(API_URL, json={"question": question, "k": 3})
        response.raise_for_status()
    except Exception as e:
        print(f"Erro ao chamar a API: {e}")
        continue

    reply_raw = response.json().get("reply", "")
    reply_clean = remove_tags(reply_raw)

    # Similaridade semântica
    emb_reply = model.encode(reply_clean, convert_to_tensor=True)
    emb_gt = model.encode(ground_truth, convert_to_tensor=True)
    similarity = util.cos_sim(emb_reply, emb_gt).item()

    # Cobertura de palavras-chave
    coverage = keyword_coverage(reply_clean, keywords)

    # Registro
    results.append({
        "question": question,
        "reply": reply_clean,
        "similarity": round(similarity, 4),
        "keyword_coverage": round(coverage, 4)
    })

    print(f"\n--- Q{i+1}: {question}")
    print(f"Similarity: {similarity:.3f} | Coverage: {coverage:.1%}")
    print(f"Reply: {reply_clean}")

# Salvar csv 

df = pd.DataFrame(results)
df.to_csv("metrics_output.csv", index=False)

# Fazer gráfico e heatmap 

def plot_heatmap(df):
    df_hm = df[["question", "similarity", "keyword_coverage"]].copy()
    df_hm.set_index("question", inplace=True)

    plt.figure(figsize=(10, 5))
    sns.heatmap(df_hm, annot=True, cmap="YlOrRd", fmt=".2f", linewidths=0.5, vmin=0, vmax=1)
    plt.title("Heatmap – Similarisade x Cobertura de palavras chave")
    plt.tight_layout()
    plt.savefig("metrics_heatmap.png")
    plt.show()

plot_heatmap(df)

def plot_metrics(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Barras de similaridade
    ax1 = sns.barplot(data=df, x="question", y="similarity", color="skyblue", label="Cosine Similarity")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Similarity / Coverage")
    plt.xticks(rotation=45, ha="right")

    # Linha de coverage
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="question", y="keyword_coverage", marker="o", color="orange", label="Keyword Coverage", ax=ax2)
    ax2.set_ylim(0, 1.05)

    # Legenda combinada
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.title("Metricas de Acurácia")
    plt.tight_layout()
    plt.savefig("metrics_plot.png")
    plt.show()

# Executar plotagems
if results:
    plot_metrics(df)
else:
    print("Nenhum resultado registrado. Verifique se a API esta ativa.")
