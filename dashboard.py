import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from rag_engine import CyberRAG

st.set_page_config(
    page_title="CyberAI Sentinel Dashboard",
    layout="wide",
    page_icon="🛡️",
)

DATA_PATH = os.path.join("data", "merged_cyber_incidents.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in df.columns:
        df[col] = df[col].fillna("")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_critical"] = (df["severity"] >= 8).astype(int)
    return df

@st.cache_resource
def load_rag():
    return CyberRAG()

@st.cache_resource
def train_rf_model(df: pd.DataFrame):
    numeric_features = [
        "asset_vulnerability_score",
        "time_to_resolve_hours",
        "data_breached_gb",
        "financial_impact_k",
        "threat_id",
        "asset_id",
        "employee_id",
    ]

    categorical_features = [
        "status",
        "threat_name",
        "threat_type",
        "asset_type",
        "asset_owner_department",
        "emp_department",
        "emp_security_training",
    ]

    X = df[numeric_features + categorical_features]
    y = df["is_critical"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            )),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model.fit(X_train, y_train)
    y_proba = rf_model.predict_proba(X)[:, 1]
    df_scores = df.copy()
    df_scores["threat_score"] = (y_proba * 100).round(1)
    return rf_model, df_scores

df = load_data()
rag = load_rag()
rf_model, df_scores = train_rf_model(df)

st.title("🛡️ CyberAI Sentinel – Cyber Analytics & RAG Assistant")

# SIDEBAR FILTERS
st.sidebar.title("Filters")
severity_min, severity_max = st.sidebar.slider(
    "Severity range", int(df["severity"].min()), int(df["severity"].max()), (1, 10)
)

departments = sorted(df["asset_owner_department"].unique())
selected_depts = st.sidebar.multiselect(
    "Asset owner departments", departments, default=departments
)

mask = (
    (df_scores["severity"] >= severity_min)
    & (df_scores["severity"] <= severity_max)
    & (df_scores["asset_owner_department"].isin(selected_depts))
)

df_filt = df_scores[mask]

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Incidents", len(df_filt))
with col2:
    st.metric("Critical Incidents", int(df_filt["is_critical"].sum()))
with col3:
    st.metric("Avg Threat Score", f"{df_filt['threat_score'].mean():.1f}")
with col4:
    st.metric(
        "Avg Time to Resolve (hrs)",
        f"{pd.to_numeric(df_filt['time_to_resolve_hours'], errors='coerce').mean():.1f}",
    )

st.markdown("---")

# MAIN VISUALS
col_left, col_right = st.columns((2, 1))

with col_left:
    st.subheader("Threat Score Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df_filt["threat_score"], bins=20, edgecolor="black")
    ax.set_xlabel("Threat Score (0–100)")
    ax.set_ylabel("Number of Incidents")
    ax.set_title("Threat Score Distribution")
    st.pyplot(fig)

    st.subheader("Attack Timeline")
    timeline_df = df_filt.copy()
    timeline_df["time_bucket"] = timeline_df["timestamp"].dt.floor("D")

    events_per_day = (
        timeline_df.groupby("time_bucket")
        .size()
        .reset_index(name="event_count")
        .sort_values("time_bucket")
    )

    crit_per_day = (
        timeline_df[timeline_df["is_critical"] == 1]
        .groupby("time_bucket")
        .size()
        .reset_index(name="critical_count")
        .sort_values("time_bucket")
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(events_per_day["time_bucket"], events_per_day["event_count"], label="All Incidents")
    ax2.plot(crit_per_day["time_bucket"], crit_per_day["critical_count"], label="Critical Incidents", linestyle="--")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Count")
    ax2.set_title("Incidents Over Time")
    ax2.legend()
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)

with col_right:
    st.subheader("Department vs Severity Heatmap")
    heat_df = df_filt[df_filt["is_critical"] == 1].copy()

    if not heat_df.empty:
        pivot = (
            heat_df.groupby(["asset_owner_department", "severity"])
            .size()
            .reset_index(name="count")
        )

        top_depts = (
            pivot.groupby("asset_owner_department")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .index
        )

        pivot_top = pivot[pivot["asset_owner_department"].isin(top_depts)]
        heatmap_data = pivot_top.pivot(
            index="asset_owner_department", columns="severity", values="count"
        ).fillna(0)

        fig3, ax3 = plt.subplots(figsize=(6, 6))
        sns.heatmap(heatmap_data, cmap="Reds", ax=ax3)
        ax3.set_xlabel("Severity")
        ax3.set_ylabel("Asset Owner Department")
        ax3.set_title("Critical Incidents – Dept vs Severity")
        st.pyplot(fig3)
    else:
        st.info("No critical incidents in current filter selection.")

st.markdown("---")

# RAG ASSISTANT
st.subheader("🤖 Cyber RAG Assistant")

query = st.text_area(
    "Ask a question about incidents, threats, or departments:",
    placeholder="e.g., Which departments are most impacted by high-severity incidents?",
)

top_k = st.slider("Number of incidents to retrieve for context", 3, 10, 5)

if st.button("Analyze with RAG"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant incidents..."):
            retrieved = rag.retrieve(query, top_k=top_k)

        st.write("### Most Relevant Incidents")
        st.dataframe(
            retrieved[
                [
                    "incident_id",
                    "timestamp",
                    "threat_name",
                    "threat_type",
                    "severity",
                    "asset_owner_department",
                    "description",
                    "similarity",
                ]
            ]
        )

        if st.checkbox("Generate natural-language answer using LLM (requires OPENAI_API_KEY)"):
            with st.spinner("Calling LLM..."):
                answer = rag.ask_llm(query, retrieved)
            st.write("### LLM Answer")
            st.write(answer)
