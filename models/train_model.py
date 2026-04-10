"""
AgriSense-AI — models/train.py
Crop Recommendation Model — Fixed for sensor_Crop_Dataset (1).csv

Columns in CSV : Nitrogen, Phosphorus, Potassium, Temperature,
                 Humidity, pH_Value, Rainfall, Crop, Soil_Type, Variety
Target         : Crop  (6 classes: Maize, Potato, Rice, Sugarcane, Tomato, Wheat)
Dropped        : Soil_Type, Variety  (not predictive sensor features)

Run:
    python models/train.py
"""

# ── Imports ───────────────────────────────────────────────────────────
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import joblib
import sys
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

matplotlib.use("Agg")


# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Column config ─────────────────────────────────────────────────────
FEATURE_COLS = ["Nitrogen", "Phosphorus", "Potassium",
                "Temperature", "Humidity", "pH_Value", "Rainfall"]
FEATURE_SHORT = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]
TARGET_COL = "Crop"
DROP_COLS = ["Soil_Type", "Variety"]

FEATURE_RANGES = {
    "Nitrogen":    (0,   200),
    "Phosphorus":  (0,   200),
    "Potassium":   (0,   200),
    "Temperature": (0,    50),
    "Humidity":    (0,   100),
    "pH_Value":    (0,    14),
    "Rainfall":    (0,   500),
}

# ── Helpers ───────────────────────────────────────────────────────────


def log(msg, level="info"):
    icons = {"ok": "✅", "warn": "⚠️ ", "step": "🔷", "info": "  "}
    print(f"  {icons.get(level, '  ')} {msg}")


def section(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


# ═════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD
# ═════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    section("STEP 1 — Load Dataset")

    candidates = [
        DATA_DIR / "crop_recommend.csv",
        DATA_DIR / "Crop_recommendation.csv",
        DATA_DIR / "crop_data.csv"
    ]

    path = next((p for p in candidates if p.exists()), None)

    if path is None:
        log(f"No CSV found in {DATA_DIR}", "warn")
        sys.exit(1)

    log(f"Loading -> {path.name}", "step")

    df = pd.read_csv(path)

    log(f"Raw shape : {df.shape}", "ok")
    log(f"Columns   : {df.columns.tolist()}", "info")

    # =========================================================
    # 🔥 FIX COLUMN NAMES (MUST BE BEFORE ANY PROCESSING)
    # =========================================================
    df = df.rename(columns={
        'phosphorus': 'Phosphorus',
        'potassium': 'Potassium',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'ph': 'pH_Value',
        'rainfall': 'Rainfall',
        'label': 'Crop'
    })

    # Remove unwanted unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Debug (optional)
    print("\n✅ Updated Columns:", df.columns.tolist())

    # =========================================================
    # CLEANING
    # =========================================================

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    if before - len(df):
        log(f"Removed {before - len(df)} duplicates", "warn")

    # Check missing values
    if df[FEATURE_COLS + [TARGET_COL]].isnull().sum().sum():
        df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
        log("Dropped rows with null values", "warn")

    # Clip out-of-range values
    for feat, (lo, hi) in FEATURE_RANGES.items():
        if feat in df.columns:
            oor = ((df[feat] < lo) | (df[feat] > hi)).sum()
            if oor:
                log(f"Clipping {oor} values in '{feat}'", "warn")
            df[feat] = df[feat].clip(lo, hi)

    # Final dataset info
    vc = df[TARGET_COL].value_counts()

    log(f"Final shape : {df.shape}", "ok")
    log(f"Crops ({df[TARGET_COL].nunique()}): {sorted(df[TARGET_COL].unique())}", "ok")
    log(f"Samples/crop: min={vc.min()}, max={vc.max()}", "ok")

    return df


# ═════════════════════════════════════════════════════════════════════
#  STEP 2 — EDA
# ═════════════════════════════════════════════════════════════════════
def plot_eda(df: pd.DataFrame):
    section("STEP 2 — EDA")
    PAL = ["#1D9E75", "#534AB7", "#D85A30", "#378ADD",
           "#BA7517", "#D4537E", "#639922"]

    # Feature distributions
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Feature Distributions", fontsize=14,
                 fontweight="bold", y=1.01)
    for i, (feat, short, col) in enumerate(
            zip(FEATURE_COLS, FEATURE_SHORT, PAL)):
        ax = axes[i // 4][i % 4]
        ax.hist(df[feat], bins=40, color=col, alpha=0.82,
                edgecolor="white", linewidth=0.4)
        ax.set_title(short, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        mu = df[feat].mean()
        ax.axvline(mu, color="black", linestyle="--",
                   linewidth=1.2, alpha=0.6)
        ax.text(mu, ax.get_ylim()[1] * 0.88,
                f" mu={mu:.1f}", fontsize=8)
    ax = axes[1][3]
    vc = df[TARGET_COL].value_counts()
    ax.barh(vc.index, vc.values,
            color=PAL[:len(vc)], alpha=0.85)
    ax.set_title("Samples per Crop", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_distributions.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log("eda_distributions.png saved", "ok")

    # Box plots per crop
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Feature per Crop — separability check",
                 fontsize=14, fontweight="bold", y=1.01)
    crop_order = sorted(df[TARGET_COL].unique())
    for i, (feat, short) in enumerate(
            zip(FEATURE_COLS, FEATURE_SHORT)):
        ax = axes[i // 4][i % 4]
        data = [df[df[TARGET_COL] == c][feat].values
                for c in crop_order]
        bp = ax.boxplot(data, patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], PAL[:len(crop_order)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(crop_order) + 1))
        ax.set_xticklabels(crop_order, rotation=35,
                           ha="right", fontsize=8)
        ax.set_title(short, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1][3].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_boxplots.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log("eda_boxplots.png saved", "ok")

    # Correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[FEATURE_COLS].corr()
    corr.index = FEATURE_SHORT
    corr.columns = FEATURE_SHORT
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_correlation.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log("eda_correlation.png saved", "ok")

    # Crop heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    crop_means = df.groupby(TARGET_COL)[FEATURE_COLS].mean()
    crop_means.columns = FEATURE_SHORT
    crop_means_norm = (
        (crop_means - crop_means.min())
        / (crop_means.max() - crop_means.min() + 1e-9)
    )
    sns.heatmap(crop_means_norm, ax=ax,
                annot=crop_means.round(1), fmt=".1f",
                cmap="YlGn", linewidths=0.5)
    ax.set_title("Mean Feature per Crop (normalised)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_crop_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log("eda_crop_heatmap.png saved", "ok")


# ═════════════════════════════════════════════════════════════════════
#  STEP 3 — PREPROCESSING
# ═════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    section("STEP 3 — Preprocessing")

    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL].values)

    # Split BEFORE scaling — no data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit scaler on train only — transform both
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)       # ← .transform() only

    log(f"Train: {X_train_sc.shape[0]:,}  |  Test: {X_test_sc.shape[0]:,}", "ok")
    log(f"Classes ({len(le.classes_)}): {list(le.classes_)}", "ok")

    return X_train_sc, X_test_sc, y_train, y_test, scaler, le


# ═════════════════════════════════════════════════════════════════════
#  STEP 4 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════
def compare_models(X_train, y_train):
    section("STEP 4 — Model Comparison (5-Fold CV)")
    candidates = {
        "Random Forest":       RandomForestClassifier(
            n_estimators=200, max_depth=15,
            class_weight="balanced",
            random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    print(f"\n  {'Model':<25} {'Mean Acc':>10}  {'Std':>8}")
    print("  " + "-" * 46)
    for name, mdl in candidates.items():
        scores = cross_val_score(
            mdl, X_train, y_train,
            cv=cv, scoring="accuracy", n_jobs=-1
        )
        results[name] = {"model": mdl,
                         "mean": scores.mean(),
                         "std":  scores.std()}
        bar = chr(9608) * int(scores.mean() * 40)
        print(f"  {name:<25} {scores.mean():.4f}     "
              f"+/-{scores.std():.4f}  {bar}")

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]
    colors = ["#1D9E75", "#534AB7", "#D85A30", "#378ADD"]
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CV Accuracy", fontsize=11)
    ax.set_title("Model Comparison — 5-Fold CV",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{val:.3f}", ha="center",
                fontsize=10, fontweight="bold")
    plt.xticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log("model_comparison.png saved", "ok")
    best = max(results, key=lambda n: results[n]["mean"])
    log(f"Best: {best}  ({results[best]['mean']:.4f})", "ok")


# ═════════════════════════════════════════════════════════════════════
#  STEP 5 — TRAIN FINAL MODEL
# ═════════════════════════════════════════════════════════════════════
def train_final(X_train, X_test, y_train, y_test, le):
    section("STEP 5 — Train Final Model")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    wf1 = f1_score(y_test, y_pred, average="weighted")
    mf1 = f1_score(y_test, y_pred, average="macro")

    log(f"Train accuracy : {train_acc:.4f}", "ok")
    log(f"Test  accuracy : {test_acc:.4f}",  "ok")
    log(f"Weighted F1    : {wf1:.4f}",        "ok")
    log(f"Macro F1       : {mf1:.4f}",        "ok")

    gap = train_acc - test_acc
    if gap > 0.10:
        log(f"Overfit gap {gap:.3f} — consider reducing max_depth", "warn")
    else:
        log(f"Overfit gap {gap:.3f} — good generalisation", "ok")

    print("\n  Classification Report:")
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, digits=3
    )
    print("\n".join("  " + line for line in report.splitlines()))
    return model, y_pred


# ═════════════════════════════════════════════════════════════════════
#  STEP 6 — CONFUSION MATRIX
# ═════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_test, y_pred, le):
    section("STEP 6 — Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, ax=ax, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(
        f"Confusion Matrix  (Test: {accuracy_score(y_test, y_pred):.2%})",
        fontsize=13, fontweight="bold"
    )
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log("confusion_matrix.png saved", "ok")


# ═════════════════════════════════════════════════════════════════════
#  STEP 7 — FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════
def plot_feature_importance(model, X_test, y_test):
    section("STEP 7 — Feature Importance")
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=15, random_state=42, n_jobs=-1
    )
    perm_df = pd.DataFrame({
        "feature":    FEATURE_SHORT,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=True)

    tree_df = pd.DataFrame({
        "feature":    FEATURE_SHORT,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1D9E75" if v >= 0 else "#D85A30"
              for v in perm_df["importance"]]
    ax1.barh(perm_df["feature"], perm_df["importance"],
             xerr=perm_df["std"], color=colors, alpha=0.85,
             error_kw=dict(elinewidth=1.2, capsize=4))
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_title("Permutation Importance",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Mean accuracy decrease")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.barh(tree_df["feature"], tree_df["importance"],
             color="#534AB7", alpha=0.82)
    ax2.set_title("Tree-Based (MDI) Importance",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Importance score")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Feature Importance — AgriSense-AI",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log("feature_importance.png saved", "ok")

    print("\n  Feature ranking:")
    for _, row in perm_df.sort_values(
            "importance", ascending=False).iterrows():
        bar = chr(9608) * max(0, int(row["importance"] * 300))
        sign = "+" if row["importance"] >= 0 else ""
        print(f"    {row['feature']:<10} "
              f"{sign}{row['importance']:.4f} +/- {row['std']:.4f}  {bar}")


# ═════════════════════════════════════════════════════════════════════
#  STEP 8 — SAVE ARTEFACTS
# ═════════════════════════════════════════════════════════════════════
def save_artefacts(model, scaler, le):
    section("STEP 8 — Save Artefacts")
    joblib.dump(model,  MODEL_DIR / "crop_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(le,     MODEL_DIR / "label_encoder.joblib")
    meta = {
        "feature_cols":   FEATURE_COLS,
        "feature_short":  FEATURE_SHORT,
        "feature_ranges": FEATURE_RANGES,
        "target_col":     TARGET_COL,
        "drop_cols":      DROP_COLS,
        "n_classes":      len(le.classes_),
        "crop_classes":   list(le.classes_),
    }
    with open(MODEL_DIR / "model_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    for fname in ["crop_model.joblib", "scaler.joblib",
                  "label_encoder.joblib", "model_metadata.json"]:
        size = (MODEL_DIR / fname).stat().st_size / 1024
        log(f"{fname:<30}  {size:.1f} KB", "ok")


# ═════════════════════════════════════════════════════════════════════
#  PREDICT FUNCTION  — used by agents/prediction_agent.py
# ═════════════════════════════════════════════════════════════════════
def predict_crop(input_dict, model=None, scaler=None,
                 le=None, top_n=3):
    errors = []
    for feat, (lo, hi) in FEATURE_RANGES.items():
        val = input_dict.get(feat)
        if val is None:
            errors.append(f"Missing: '{feat}'")
        elif not (lo <= float(val) <= hi):
            errors.append(
                f"'{feat}'={val} out of range [{lo}, {hi}]"
            )
    if errors:
        return {"error": errors, "input_validated": False}

    if model is None:
        model = joblib.load(MODEL_DIR / "crop_model.joblib")
        scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        le = joblib.load(MODEL_DIR / "label_encoder.joblib")

    x_sc = scaler.transform(
        np.array([[input_dict[f] for f in FEATURE_COLS]])
    )
    proba = model.predict_proba(x_sc)[0]
    top_idx = proba.argsort()[-top_n:][::-1]

    return {
        "top_crops": [
            {
                "rank":        rank + 1,
                "crop":        le.classes_[i],
                "confidence":  round(proba[i] * 100, 2),
                "probability": round(float(proba[i]), 4),
            }
            for rank, i in enumerate(top_idx)
        ],
        "feature_importance": dict(zip(
            FEATURE_COLS,
            [round(float(v), 4) for v in model.feature_importances_]
        )),
        "input_validated": True,
    }


# ═════════════════════════════════════════════════════════════════════
#  STEP 9 — DEMO PREDICTIONS
# ═════════════════════════════════════════════════════════════════════
def demo_predictions(model, scaler, le, df):
    section("STEP 9 — Sample Predictions (Smoke Test)")
    sample_list = []
    for crop_name, group in df.groupby(TARGET_COL):
        row = group.sample(1, random_state=42).copy()
        sample_list.append(row)
    samples = pd.concat(sample_list).reset_index(drop=True)

    correct = 0
    for _, row in samples.iterrows():
        inp = {f: row[f] for f in FEATURE_COLS}
        true_label = row[TARGET_COL]
        result = predict_crop(inp, model, scaler, le, top_n=3)
        if "error" in result:
            log(f"Validation error: {result['error']}", "warn")
            continue
        top = result["top_crops"]
        ok = "OK" if top[0]["crop"] == true_label else "WRONG"
        if top[0]["crop"] == true_label:
            correct += 1
        print(f"\n  N={inp['Nitrogen']:.1f} P={inp['Phosphorus']:.1f} "
              f"K={inp['Potassium']:.1f} T={inp['Temperature']:.1f}C "
              f"H={inp['Humidity']:.1f}% pH={inp['pH_Value']:.1f} "
              f"Rain={inp['Rainfall']:.1f}mm")
        print(f"  True -> {true_label}  [{ok}]")
        for t in top:
            arrow = " <- predicted" if t["rank"] == 1 else ""
            bar = chr(9619) * int(t["confidence"] / 3)
            print(f"    #{t['rank']} {t['crop']:<14} "
                  f"{t['confidence']:5.1f}%  {bar}{arrow}")
    print(f"\n  Score: {correct}/{len(samples)} correct")


# ═════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════
def train_pipeline():
    print("\n" + "=" * 63)
    print("     AgriSense-AI -- Crop Recommendation Training")
    print("=" * 63)

    df = load_data()
    plot_eda(df)
    X_train, X_test, y_train, y_test, \
        scaler, le = preprocess(df)
    compare_models(X_train, y_train)
    model, y_pred = train_final(
        X_train, X_test,
        y_train, y_test, le)
    plot_confusion_matrix(y_test, y_pred, le)
    plot_feature_importance(model, X_test, y_test)
    save_artefacts(model, scaler, le)
    demo_predictions(model, scaler, le, df)

    section("TRAINING COMPLETE")
    test_acc = accuracy_score(y_test, y_pred)
    log(f"Test Accuracy : {test_acc:.4f}  ({test_acc * 100:.1f}%)", "ok")
    log("Artefacts     -> models/", "ok")
    log("Charts        -> outputs/", "ok")
    log("Next step     -> Phase 3: RAG Pipeline", "ok")
    print()


if __name__ == "__main__":
    train_pipeline()
