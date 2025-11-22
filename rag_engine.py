import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

TEXT_PATH = os.path.join("data", "incident_text.parquet")
EMB_PATH = os.path.join("data", "incident_embeddings.npy")

class CyberRAG:
    def __init__(self):
        print("[RAG] Loading incident text & embeddings...")
        self.df = pd.read_parquet(TEXT_PATH)
        self.embeddings = np.load(EMB_PATH)
        print(f"[RAG] Loaded {len(self.df)} incidents.")

        print("[RAG] Loading embedding model (for query encoding)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        results = self.df.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]
        return results

    def build_context(self, retrieved_df: pd.DataFrame) -> str:
        lines = []
        for _, row in retrieved_df.iterrows():
            lines.append(
                f"Incident {row.get('incident_id', '')} | "
                f"Time: {row.get('timestamp', '')} | "
                f"Threat: {row.get('threat_name', '')} ({row.get('threat_type', '')}) | "
                f"Severity: {row.get('severity', '')} | "
                f"Department: {row.get('asset_owner_department', '')} | "
                f"Description: {row.get('description', '')}"
            )
        return "\n".join(lines)

    def ask_llm(self, query: str, retrieved_df: pd.DataFrame) -> str:
        if not OPENAI_AVAILABLE:
            return "OpenAI is not installed. Install it with `pip install openai`."

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEY not set. Please set it in your environment."

        openai.api_key = api_key
        context = self.build_context(retrieved_df)

        prompt = f"""
You are a cybersecurity analyst assistant.
Use ONLY the incident context below to answer the analyst's question.
Explain clearly, reference relevant incidents, and suggest next steps.

Incident context:
{context}

Analyst question:
{query}
"""

        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful cybersecurity analyst assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
