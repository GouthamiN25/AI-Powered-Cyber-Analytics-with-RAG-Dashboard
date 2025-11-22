"""
prepare_rag.py
Builds RAG-ready text and embeddings for CyberAI Sentinel.
Run once after updating merged_cyber_incidents.csv
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = os.path.join("data", "merged_cyber_incidents.csv")
TEXT_OUT_PATH = os.path.join("data", "incident_text.parquet")
EMB_OUT_PATH = os.path.join("data", "incident_embeddings.npy")

def load_data():
    abs_path = os.path.abspath(DATA_PATH)
    print(f"[INFO] Loading data from: {abs_path}")
    df = pd.read_csv(abs_path)
    print(f"[INFO] Shape: {df.shape}")
    return df

def build_rag_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a 'rag_text' column combining useful fields for semantic search.
    Adjust column names here if your CSV differs.
    """
    df = df.copy()

    # Fill NaNs to avoid 'nan' strings
    for col in df.columns:
        df[col] = df[col].fillna("")

    def row_to_text(row):
        parts = [
            f"Incident ID: {row.get('incident_id', '')}",
            f"Time: {row.get('timestamp', '')}",
            f"Threat: {row.get('threat_name', '')} ({row.get('threat_type', '')})",
            f"Severity: {row.get('severity', '')}",
            f"Status: {row.get('status', '')}",
            f"Asset: {row.get('asset_name', '')} ({row.get('asset_type', '')})",
            f"Department: {row.get('asset_owner_department', '')}",
            f"Employee department: {row.get('emp_department', '')}",
            f"Vulnerability score: {row.get('asset_vulnerability_score', '')}",
            f"Time to resolve (hours): {row.get('time_to_resolve_hours', '')}",
            f"Data breached (GB): {row.get('data_breached_gb', '')}",
            f"Financial impact (k): {row.get('financial_impact_k', '')}",
            f"Description: {row.get('description', '')}",
        ]
        return " | ".join([p for p in parts if p.strip()])

    print("[INFO] Building RAG text for each incident...")
    df["rag_text"] = [row_to_text(r) for _, r in tqdm(df.iterrows(), total=len(df))]
    return df

def compute_embeddings(texts):
    print("[INFO] Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    return embeddings

def main():
    df = load_data()
    df_rag = build_rag_text(df)

    # Save text + metadata
    df_rag.to_parquet(TEXT_OUT_PATH, index=False)
    print(f"[INFO] Saved incident text & metadata to {TEXT_OUT_PATH}")

    # Compute and save embeddings
    embeddings = compute_embeddings(df_rag["rag_text"].tolist())
    np.save(EMB_OUT_PATH, embeddings)
    print(f"[INFO] Saved embeddings to {EMB_OUT_PATH}")

if __name__ == "__main__":
    main()
