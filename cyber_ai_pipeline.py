"""
CyberAI Sentinel - AI-Powered Cyber Analytics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# ---------- CONFIG ----------
DATA_PATH = os.path.join("data", "merged_cyber_incidents.csv")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------- 1. LOAD DATA ----------
def load_data(path: str) -> pd.DataFrame:
    abs_path = os.path.abspath(path)
    print(f"[INFO] Trying to load data from: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"[ERROR] CSV not found at: {abs_path}")

    df = pd.read_csv(abs_path)
    print(f"[INFO] Loaded data shape: {df.shape}")
    return df



# ---------- 2. PREPROCESSING ----------
def prepare_data(df: pd.DataFrame):
    """
    Create a binary target from severity:
    is_critical = 1 if severity >= 8 else 0
    And build feature matrix X with relevant columns.
    """

    # Create binary target
    df = df.copy()
    df["is_critical"] = (df["severity"] >= 8).astype(int)

    # Keep severity for later analysis/plots, but NOT as a feature
    target = "is_critical"

    # Choose features (drop IDs and free-text fields like description, names)
    feature_cols_numeric = [
        "asset_vulnerability_score",
        "time_to_resolve_hours",
        "data_breached_gb",
        "financial_impact_k",
        "threat_id",
        "asset_id",
        "employee_id",
    ]

    feature_cols_categorical = [
        "status",
        "threat_name",
        "threat_type",
        "asset_type",
        "asset_owner_department",
        "emp_department",
        "emp_security_training",
    ]

    # For timeline & heatmap later
    aux_cols = [
        "timestamp",
        "severity",
        "asset_owner_department",
    ]

    # Subset
    X = df[feature_cols_numeric + feature_cols_categorical]
    y = df[target]

    aux = df[aux_cols].copy()

    return X, y, aux, feature_cols_numeric, feature_cols_categorical


def build_preprocessor(numeric_features, categorical_features):
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
    return preprocessor


# ---------- 3. MODEL TRAINING ----------
def train_models(X_train, y_train, preprocessor):
    # Logistic Regression (baseline)
    log_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    # Random Forest (main model)
    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )),
        ]
    )

    print("[INFO] Training Logistic Regression...")
    log_reg.fit(X_train, y_train)

    print("[INFO] Training Random Forest...")
    rf_model.fit(X_train, y_train)

    return log_reg, rf_model


# ---------- 4. EVALUATION ----------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: use decision_function if available, else zeros
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # normalize to 0-1 range
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        else:
            y_proba = np.zeros_like(y_test, dtype=float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n===== {name} Evaluation =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Critical"]))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"roc_curve_{name.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] ROC curve saved to {out_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Critical"])

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {out_path}")

    return y_pred, y_proba


# ---------- 5. VISUALIZATIONS ----------
def plot_feature_importance(rf_model, numeric_features, categorical_features):
    """
    Extract feature importances from Random Forest.
    rf_model is the Pipeline(preprocess + model).
    """

    print("[INFO] Plotting feature importance for Random Forest...")

    # Get trained pieces
    preprocessor = rf_model.named_steps["preprocess"]
    model = rf_model.named_steps["model"]

    # Get feature names after one-hot encoding
    cat_transformer = preprocessor.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["encoder"]

    cat_feature_names = encoder.get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, cat_feature_names])

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_n = min(20, len(importances))  # show top 20
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance – Random Forest")
    plt.bar(range(top_n), importances[indices][:top_n])
    plt.xticks(range(top_n), feature_names[indices][:top_n], rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "feature_importance_rf.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Feature importance plot saved to {out_path}")


def plot_threat_score_distribution(y_proba_rf):
    print("[INFO] Plotting threat score distribution...")
    threat_scores = (y_proba_rf * 100).round(1)

    plt.figure(figsize=(8, 5))
    plt.hist(threat_scores, bins=20, edgecolor="black")
    plt.xlabel("Threat Score (0–100)")
    plt.ylabel("Number of Incidents")
    plt.title("Threat Score Distribution – Random Forest")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "threat_score_distribution_rf.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Threat score distribution saved to {out_path}")


def plot_department_severity_heatmap(aux_test, y_test, y_pred):
    """
    Equivalent to 'IP vs severity heatmap' but using asset_owner_department vs severity
    since this dataset has no IP addresses.
    """

    print("[INFO] Plotting Department vs Severity heatmap...")

    df = aux_test.copy()
    df["true_label"] = y_test.values
    df["pred_label"] = y_pred

    # Optionally focus on predicted critical incidents
    df_crit = df[df["pred_label"] == 1]

    # Group by department & severity
    pivot = (
        df_crit.groupby(["asset_owner_department", "severity"])
        .size()
        .reset_index(name="count")
    )

    # Limit to top departments for readability
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

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=False, cmap="Reds")
    plt.title("Predicted Critical Incidents – Department vs Severity")
    plt.xlabel("Severity")
    plt.ylabel("Asset Owner Department")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "department_severity_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Department vs Severity heatmap saved to {out_path}")


def plot_attack_timeline(aux_test, y_test, y_pred):
    print("[INFO] Plotting attack timeline...")

    df = aux_test.copy()
    df["true_label"] = y_test.values
    df["pred_label"] = y_pred

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Time bucket (e.g., by day)
    df["time_bucket"] = df["timestamp"].dt.floor("D")

    events_per_bucket = (
        df.groupby("time_bucket")
        .size()
        .reset_index(name="event_count")
        .sort_values("time_bucket")
    )

    crit_per_bucket = (
        df[df["pred_label"] == 1]
        .groupby("time_bucket")
        .size()
        .reset_index(name="critical_count")
        .sort_values("time_bucket")
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        events_per_bucket["time_bucket"],
        events_per_bucket["event_count"],
        label="All Incidents",
    )
    plt.plot(
        crit_per_bucket["time_bucket"],
        crit_per_bucket["critical_count"],
        label="Predicted Critical Incidents",
        linestyle="--",
    )
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.title("Attack Timeline – Incidents Over Time")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "attack_timeline.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Attack timeline saved to {out_path}")


# ---------- MAIN ----------
def main():
    # Load
    df = load_data(DATA_PATH)

    # Prepare
    X, y, aux, num_features, cat_features = prepare_data(df)
    preprocessor = build_preprocessor(num_features, cat_features)

    # Split
    X_train, X_test, y_train, y_test, aux_train, aux_test = train_test_split(
        X, y, aux, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Train models
    log_reg, rf_model = train_models(X_train, y_train, preprocessor)

    # Evaluate models
    y_pred_lr, y_proba_lr = evaluate_model("Logistic Regression", log_reg, X_test, y_test)
    y_pred_rf, y_proba_rf = evaluate_model("Random Forest", rf_model, X_test, y_test)

    # Visualizations (using Random Forest as main model)
    plot_feature_importance(rf_model, num_features, cat_features)
    plot_threat_score_distribution(y_proba_rf)
    plot_department_severity_heatmap(aux_test, y_test, y_pred_rf)
    plot_attack_timeline(aux_test, y_test, y_pred_rf)

    print("\n[INFO] Pipeline complete. Check the 'results/' folder for plots.")


if __name__ == "__main__":
    main()
