"""
=============================================================================
INTELLIGENT DIABETES RISK PREDICTION AND PERSONALIZED NUTRITION SYSTEM
=============================================================================
Hybrid Approach: Artificial Neural Network (ANN) + Fuzzy Logic

DATASET: PIMA Indians Diabetes Dataset
Source  : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
File    : diabetes.csv  (place in the same directory as this script)

USAGE:
    1. Download diabetes.csv from the Kaggle link above.
    2. Place it in the same folder as this script  — OR update DATASET_PATH below.
    3. Run:  python diabetes_system.py
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0.  CONFIGURATION
# ---------------------------------------------------------------------------
DATASET_PATH = r"C:\Users\cherisma\Downloads\diabetes\diabetes.csv"   # <-- UPDATE THIS PATH IF NEEDED
GRAPHS_DIR   = r"C:\Users\cherisma\Downloads\SC\graphs"
RANDOM_SEED  = 42

# ---------------------------------------------------------------------------
# 1.  IMPORTS
# ---------------------------------------------------------------------------
import os, warnings, sys
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["KERAS_BACKEND"]          = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection  import StratifiedKFold, train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score,
                                      confusion_matrix, roc_curve, auc,
                                      classification_report)
from sklearn.utils            import class_weight

import keras
from keras import Sequential
from keras.layers      import Dense, Dropout, BatchNormalization
from keras.optimizers  import Adam
from keras.regularizers import l2
from keras.callbacks   import EarlyStopping, ReduceLROnPlateau

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------------------------------------------------------
# 2.  SEEDS & DIRS
# ---------------------------------------------------------------------------
np.random.seed(RANDOM_SEED)
keras.utils.set_random_seed(RANDOM_SEED)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ============================================================================
# MODULE A — DATA LOADING & PREPROCESSING
# ============================================================================
def load_and_preprocess(path: str):
    if not os.path.exists(path):
        sys.exit(
            f"\n[ERROR] Dataset not found at '{path}'.\n"
            "Download diabetes.csv from:\n"
            "  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
            "and place it in the same directory as this script.\n"
        )

    df = pd.read_csv(path)
    print(f"\n{'='*65}")
    print("  DIABETES RISK PREDICTION & PERSONALIZED NUTRITION SYSTEM")
    print(f"{'='*65}")
    print(f"\n[DATA]  Loaded '{path}'  ->  {df.shape[0]} rows, {df.shape[1]} cols")

    # Replace physiologically impossible zeros with NaN
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid:
        df[col] = df[col].replace(0, np.nan)

    # Class-stratified median imputation
    for col in zero_invalid:
        df[col] = df.groupby("Outcome")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # ---- Derived features ----
    df["PhysicalActivity"] = (
        100 - df["Glucose"].clip(upper=200) * 0.30
            - df["BMI"].clip(upper=60)     * 0.50
    ).clip(0, 100)

    df["DietScore"] = (
        100 - df["Glucose"].clip(upper=200) * 0.35
            - df["BMI"].clip(upper=60)     * 0.65
    ).clip(0, 100)

    df["StressLevel"] = (
        (df["Insulin"].clip(upper=500)       / 500) * 50 +
        (df["BloodPressure"].clip(upper=122) / 122) * 50
    ).clip(0, 100)

    feature_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        "PhysicalActivity", "DietScore", "StressLevel"
    ]
    X = df[feature_cols].values
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[DATA]  Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
    print(f"[DATA]  Class dist (train) -- No Diab: {(y_train==0).sum()} "
          f"| Diab: {(y_train==1).sum()}")

    return X_train, X_test, y_train, y_test, scaler, df, feature_cols

# ============================================================================
# MODULE B — ANN MODEL
# ============================================================================
def build_ann(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,),
              kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.30),

        Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.25),

        Dense(64,  activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.20),

        Dense(32,  activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.15),

        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_ann(X_train, y_train, X_test, y_test):
    cw      = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    cw_dict = {0: cw[0], 1: cw[1]}

    n_splits = 5
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=RANDOM_SEED)
    oof_probs  = np.zeros(len(y_train))
    test_probs = np.zeros(len(y_test))
    histories  = []

    print(f"\n[ANN]  Training {n_splits}-fold ensemble ...")

    last_model = None
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        Xtr, Xval = X_train[tr_idx], X_train[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        mdl = build_ann(X_train.shape[1])
        cbs = [
            EarlyStopping(monitor="val_loss", patience=25,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=10, min_lr=1e-6, verbose=0),
        ]
        hist = mdl.fit(
            Xtr, ytr,
            validation_data=(Xval, yval),
            epochs=350, batch_size=16,
            class_weight=cw_dict,
            callbacks=cbs,
            verbose=0
        )
        oof_probs[val_idx] = mdl.predict(Xval,   verbose=0).ravel()
        test_probs        += mdl.predict(X_test,  verbose=0).ravel() / n_splits
        histories.append(hist.history)
        last_model = mdl

        vacc = accuracy_score(yval, (oof_probs[val_idx] >= 0.5).astype(int))
        print(f"         Fold {fold}/{n_splits}  val_acc = {vacc:.4f}")

    # Optimise threshold on OOF predictions
    thresholds  = np.linspace(0.20, 0.80, 120)
    best_thresh = max(thresholds,
                      key=lambda t: f1_score(y_train,
                                             (oof_probs >= t).astype(int)))
    y_pred = (test_probs >= best_thresh).astype(int)
    print(f"\n[ANN]  Optimal threshold : {best_thresh:.3f}")

    return last_model, y_pred, test_probs, histories, best_thresh

# ============================================================================
# MODULE C — METRICS
# ============================================================================
def evaluate(y_test, y_pred, y_prob):
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, zero_division=0)
    rec    = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    rocauc = auc(fpr, tpr)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*65}")
    print("  ANN PERFORMANCE METRICS")
    print(f"{'='*65}")
    print(f"  Accuracy              : {acc*100:.2f} %")
    print(f"  Precision             : {prec*100:.2f} %")
    print(f"  Recall (Sensitivity)  : {rec*100:.2f} %")
    print(f"  F1 Score              : {f1*100:.2f} %")
    print(f"  ROC-AUC               : {rocauc:.4f}")
    print(f"\n  Full Classification Report:\n")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Diabetes", "Diabetes"]))
    return acc, prec, rec, f1, rocauc, fpr, tpr, cm

# ============================================================================
# MODULE D — FUZZY LOGIC SYSTEM
# ============================================================================

KNOWLEDGE_BASE = [
    {"id":  1, "risk":"High",   "activity":"Low",      "sugar":"High",    "rec":"Strict Diet + Intense Exercise"},
    {"id":  2, "risk":"High",   "activity":"Low",      "sugar":"Moderate","rec":"Strict Diet + Moderate Exercise"},
    {"id":  3, "risk":"High",   "activity":"Moderate", "sugar":"High",    "rec":"Strict Diet + Daily Walks"},
    {"id":  4, "risk":"High",   "activity":"Moderate", "sugar":"Moderate","rec":"Controlled Diet + Regular Exercise"},
    {"id":  5, "risk":"High",   "activity":"High",     "sugar":"High",    "rec":"Sugar-Free Diet + Maintain Exercise"},
    {"id":  6, "risk":"High",   "activity":"High",     "sugar":"Moderate","rec":"Low-Carb Diet + Maintain Activity"},
    {"id":  7, "risk":"High",   "activity":"Low",      "sugar":"Low",     "rec":"Balanced Diet + Increase Exercise"},
    {"id":  8, "risk":"High",   "activity":"High",     "sugar":"Low",     "rec":"Maintain Habits + Regular Monitoring"},
    {"id":  9, "risk":"High",   "activity":"Moderate", "sugar":"Low",     "rec":"Strict Carb Limit + Regular Exercise"},
    {"id": 10, "risk":"Medium", "activity":"Low",      "sugar":"High",    "rec":"Moderate Diet Control + Walk Daily"},
    {"id": 11, "risk":"Medium", "activity":"Low",      "sugar":"Moderate","rec":"Balanced Diet + Light Exercise"},
    {"id": 12, "risk":"Medium", "activity":"Moderate", "sugar":"High",    "rec":"Reduce Sugar Intake + Stay Active"},
    {"id": 13, "risk":"Medium", "activity":"Moderate", "sugar":"Moderate","rec":"Healthy Lifestyle + Routine Checkups"},
    {"id": 14, "risk":"Medium", "activity":"High",     "sugar":"High",    "rec":"Cut Sugar + Continue Exercise"},
    {"id": 15, "risk":"Medium", "activity":"High",     "sugar":"Moderate","rec":"Maintain Habits + Periodic Screening"},
    {"id": 16, "risk":"Medium", "activity":"Low",      "sugar":"Low",     "rec":"Increase Activity + Balanced Nutrition"},
    {"id": 17, "risk":"Medium", "activity":"High",     "sugar":"Low",     "rec":"Keep Routine + Annual Screening"},
    {"id": 18, "risk":"Medium", "activity":"Moderate", "sugar":"Low",     "rec":"Good Progress + Stay Consistent"},
    {"id": 19, "risk":"Low",    "activity":"Low",      "sugar":"High",    "rec":"Preventive Diet + Increase Activity"},
    {"id": 20, "risk":"Low",    "activity":"Low",      "sugar":"Moderate","rec":"Healthy Habits + Light Exercise"},
    {"id": 21, "risk":"Low",    "activity":"Moderate", "sugar":"High",    "rec":"Reduce Sweets + Stay Active"},
    {"id": 22, "risk":"Low",    "activity":"Moderate", "sugar":"Moderate","rec":"Maintain Healthy Lifestyle"},
    {"id": 23, "risk":"Low",    "activity":"High",     "sugar":"High",    "rec":"Monitor Sugar + Continue Exercise"},
    {"id": 24, "risk":"Low",    "activity":"High",     "sugar":"Moderate","rec":"Excellent Habits -- Keep It Up!"},
    {"id": 25, "risk":"Low",    "activity":"Low",      "sugar":"Low",     "rec":"Increase Activity + Routine Check"},
    {"id": 26, "risk":"Low",    "activity":"High",     "sugar":"Low",     "rec":"Optimal Health -- Annual Check Only"},
    {"id": 27, "risk":"Low",    "activity":"Moderate", "sugar":"Low",     "rec":"Healthy Path -- Maintain Routine"},
]


def print_knowledge_base():
    print(f"\n{'='*65}")
    print("  FUZZY LOGIC KNOWLEDGE BASE  (27 Rules)")
    print(f"{'='*65}")
    print(f"{'ID':>3} | {'Risk':<8} | {'Activity':<10} | {'Sugar':<10} | Recommendation")
    print("-" * 78)
    for r in KNOWLEDGE_BASE:
        print(f"{r['id']:>3} | {r['risk']:<8} | {r['activity']:<10} | "
              f"{r['sugar']:<10} | {r['rec']}")
    print()


def build_fuzzy_system():
    U = np.arange(0, 101, 1)

    risk     = ctrl.Antecedent(U, "DiabetesRisk")
    activity = ctrl.Antecedent(U, "ActivityLevel")
    sugar    = ctrl.Antecedent(U, "SugarIntake")
    rec_out  = ctrl.Consequent(U, "RecommendationScore",
                               defuzzify_method="centroid")

    risk["Low"]    = fuzz.trapmf(U, [0,  0,  30, 50])
    risk["Medium"] = fuzz.trimf(U,  [30, 50, 70])
    risk["High"]   = fuzz.trapmf(U, [50, 70, 100, 100])

    activity["Low"]      = fuzz.trapmf(U, [0,  0,  25, 45])
    activity["Moderate"] = fuzz.trimf(U,  [25, 50, 75])
    activity["High"]     = fuzz.trapmf(U, [55, 75, 100, 100])

    sugar["Low"]      = fuzz.trapmf(U, [0,  0,  25, 45])
    sugar["Moderate"] = fuzz.trimf(U,  [25, 50, 75])
    sugar["High"]     = fuzz.trapmf(U, [55, 75, 100, 100])

    rec_out["Maintain"] = fuzz.trapmf(U, [0,  0,  25, 40])
    rec_out["Moderate"] = fuzz.trimf(U,  [30, 50, 70])
    rec_out["Strict"]   = fuzz.trapmf(U, [60, 75, 100, 100])

    R = [
        ctrl.Rule(risk["High"]   & activity["Low"]      & sugar["High"],    rec_out["Strict"]),
        ctrl.Rule(risk["High"]   & activity["Low"]      & sugar["Moderate"],rec_out["Strict"]),
        ctrl.Rule(risk["High"]   & activity["Moderate"] & sugar["High"],    rec_out["Strict"]),
        ctrl.Rule(risk["High"]   & activity["Moderate"] & sugar["Moderate"],rec_out["Strict"]),
        ctrl.Rule(risk["High"]   & activity["High"]     & sugar["High"],    rec_out["Strict"]),
        ctrl.Rule(risk["High"]   & activity["High"]     & sugar["Moderate"],rec_out["Strict"]),
        ctrl.Rule(risk["High"]   & activity["Low"]      & sugar["Low"],     rec_out["Strict"]),
        ctrl.Rule(risk["High"]   & activity["High"]     & sugar["Low"],     rec_out["Moderate"]),
        ctrl.Rule(risk["High"]   & activity["Moderate"] & sugar["Low"],     rec_out["Strict"]),
        ctrl.Rule(risk["Medium"] & activity["Low"]      & sugar["High"],    rec_out["Strict"]),
        ctrl.Rule(risk["Medium"] & activity["Low"]      & sugar["Moderate"],rec_out["Moderate"]),
        ctrl.Rule(risk["Medium"] & activity["Moderate"] & sugar["High"],    rec_out["Moderate"]),
        ctrl.Rule(risk["Medium"] & activity["Moderate"] & sugar["Moderate"],rec_out["Moderate"]),
        ctrl.Rule(risk["Medium"] & activity["High"]     & sugar["High"],    rec_out["Moderate"]),
        ctrl.Rule(risk["Medium"] & activity["High"]     & sugar["Moderate"],rec_out["Moderate"]),
        ctrl.Rule(risk["Medium"] & activity["Low"]      & sugar["Low"],     rec_out["Moderate"]),
        ctrl.Rule(risk["Medium"] & activity["High"]     & sugar["Low"],     rec_out["Maintain"]),
        ctrl.Rule(risk["Medium"] & activity["Moderate"] & sugar["Low"],     rec_out["Moderate"]),
        ctrl.Rule(risk["Low"]    & activity["Low"]      & sugar["High"],    rec_out["Moderate"]),
        ctrl.Rule(risk["Low"]    & activity["Low"]      & sugar["Moderate"],rec_out["Maintain"]),
        ctrl.Rule(risk["Low"]    & activity["Moderate"] & sugar["High"],    rec_out["Moderate"]),
        ctrl.Rule(risk["Low"]    & activity["Moderate"] & sugar["Moderate"],rec_out["Maintain"]),
        ctrl.Rule(risk["Low"]    & activity["High"]     & sugar["High"],    rec_out["Moderate"]),
        ctrl.Rule(risk["Low"]    & activity["High"]     & sugar["Moderate"],rec_out["Maintain"]),
        ctrl.Rule(risk["Low"]    & activity["Low"]      & sugar["Low"],     rec_out["Maintain"]),
        ctrl.Rule(risk["Low"]    & activity["High"]     & sugar["Low"],     rec_out["Maintain"]),
        ctrl.Rule(risk["Low"]    & activity["Moderate"] & sugar["Low"],     rec_out["Maintain"]),
    ]

    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(R))
    return sim, risk, activity, sugar, rec_out


def fuzzy_recommend(sim, risk_score, activity_score, sugar_score):
    sim.input["DiabetesRisk"]  = float(np.clip(risk_score,     0, 100))
    sim.input["ActivityLevel"] = float(np.clip(activity_score, 0, 100))
    sim.input["SugarIntake"]   = float(np.clip(sugar_score,    0, 100))
    sim.compute()
    score = sim.output["RecommendationScore"]

    if score >= 60:
        level  = "STRICT"
        advice = [
            "  * Eliminate refined sugars & white carbohydrates completely",
            "  * Follow a calorie-deficit diet (consult a registered dietitian)",
            "  * Perform 45-60 min aerobic exercise daily",
            "  * Monitor blood glucose twice daily (fasting + post-meal)",
            "  * Consult an endocrinologist immediately",
            "  * Target HbA1c < 7 % within 3 months",
        ]
    elif score >= 30:
        level  = "MODERATE"
        advice = [
            "  * Reduce sugar and ultra-processed food intake by 50 %",
            "  * Balanced plate: 50% vegs, 25% protein, 25% whole grains",
            "  * Walk 30 min/day; add 2 strength sessions per week",
            "  * Monitor blood glucose monthly",
            "  * Physician check-up every 6 months",
            "  * Stay hydrated (>= 2 litres of water/day)",
        ]
    else:
        level  = "MAINTAIN"
        advice = [
            "  * Continue your current healthy dietary habits",
            "  * Maintain >= 150 min moderate physical activity per week",
            "  * Annual blood glucose screening",
            "  * Increase dietary fibre (fruits, vegetables, legumes)",
            "  * Limit alcohol; avoid smoking",
            "  * Keep BMI within 18.5-24.9 range",
        ]
    return level, advice, score

# ============================================================================
# MODULE E — GRAPHS
# ============================================================================
C_BLUE   = "#2563EB"; C_RED    = "#DC2626"; C_GREEN  = "#16A34A"
C_ORANGE = "#EA580C"; C_PURPLE = "#7C3AED"; C_TEAL   = "#0D9488"
BG = "#F8FAFC"


def _save(fig, name):
    p = os.path.join(GRAPHS_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAPH] Saved -> {p}")


def plot_confusion_matrix(cm):
    labels = ["No Diabetes", "Diabetes"]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="Actual",
           title="Confusion Matrix — ANN Classifier")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=22, fontweight="bold",
                    color="white" if cm[i,j] > thresh else "black")
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


def plot_roc(fpr, tpr, rocauc):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(fpr, tpr, lw=2.5, color=C_BLUE,
            label=f"ANN  (AUC = {rocauc:.4f})")
    ax.plot([0,1],[0,1], lw=1.5, linestyle="--", color="grey",
            label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.12, color=C_BLUE)
    ax.set(xlim=[0,1], ylim=[0,1.02],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — ANN Diabetes Prediction")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "roc_curve.png")


def plot_metrics_bar(acc, prec, rec, f1, rocauc):
    names  = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    vals   = [acc, prec, rec, f1, rocauc]
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, C_TEAL]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.bar(names, [v*100 for v in vals], color=colors,
                  width=0.55, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.7,
                f"{v*100:.2f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#1E293B")
    ax.axhline(95, color=C_RED, linestyle="--", lw=1.8, label="95% target")
    ax.set(ylim=[70, 107], ylabel="Score (%)",
           title="ANN Model — Performance Metrics")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "performance_metrics.png")


def plot_training_history(histories):
    mlen = max(len(h["accuracy"]) for h in histories)

    def pad(lst):
        a = np.array(lst)
        return np.pad(a, (0, mlen - len(a)), mode="edge")

    avg_acc   = np.mean([pad(h["accuracy"])     for h in histories], axis=0)
    avg_val   = np.mean([pad(h["val_accuracy"]) for h in histories], axis=0)
    avg_loss  = np.mean([pad(h["loss"])         for h in histories], axis=0)
    avg_vloss = np.mean([pad(h["val_loss"])     for h in histories], axis=0)
    E = np.arange(1, mlen + 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    for ax in (a1, a2):
        ax.set_facecolor(BG)

    a1.plot(E, avg_acc, color=C_BLUE, lw=2, label="Train")
    a1.plot(E, avg_val, color=C_RED,  lw=2, linestyle="--", label="Validation")
    a1.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy (5-Fold Average)")
    a1.legend(); a1.grid(alpha=0.3)
    a1.spines[["top","right"]].set_visible(False)

    a2.plot(E, avg_loss,  color=C_BLUE, lw=2, label="Train")
    a2.plot(E, avg_vloss, color=C_RED,  lw=2, linestyle="--", label="Validation")
    a2.set(xlabel="Epoch", ylabel="Binary Cross-Entropy",
           title="Loss (5-Fold Average)")
    a2.legend(); a2.grid(alpha=0.3)
    a2.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "training_history.png")


def plot_feature_importance(X_train, y_train, feature_cols):
    corr  = [abs(np.corrcoef(X_train[:,i], y_train)[0,1])
             for i in range(len(feature_cols))]
    order  = np.argsort(corr)
    med    = np.median(corr)
    colors = [C_RED if corr[i] >= med else C_BLUE for i in order]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh([feature_cols[i] for i in order],
            [corr[i] for i in order],
            color=colors, edgecolor="white")
    ax.set(xlabel="|Pearson Correlation with Outcome|",
           title="Feature Importance (Correlation-Based)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "feature_importance.png")


def plot_fuzzy_mf(risk_v, act_v, sug_v, rec_v):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=BG)
    fig.suptitle("Fuzzy Membership Functions", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    configs = [
        (risk_v, "Diabetes Risk",          {"Low": C_GREEN, "Medium": C_ORANGE, "High": C_RED}),
        (act_v,  "Physical Activity Level", {"Low": C_RED, "Moderate": C_ORANGE, "High": C_GREEN}),
        (sug_v,  "Sugar Intake Level",      {"Low": C_GREEN, "Moderate": C_ORANGE, "High": C_RED}),
        (rec_v,  "Recommendation Score",    {"Maintain": C_GREEN, "Moderate": C_ORANGE, "Strict": C_RED}),
    ]

    for ax, (var, title, cmap) in zip(axes, configs):
        ax.set_facecolor(BG)
        for label, term in var.terms.items():
            ax.plot(var.universe, term.mf, lw=2.5,
                    color=cmap.get(label, C_BLUE), label=label)
            ax.fill_between(var.universe, term.mf, alpha=0.12,
                            color=cmap.get(label, C_BLUE))
        ax.set(title=title, xlabel="Universe (0-100)",
               ylabel="Membership Degree", ylim=[-0.05, 1.1])
        ax.legend(fontsize=9); ax.grid(alpha=0.25)
        ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fuzzy_membership_functions.png")


def plot_risk_distribution(df):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    for ax in (a1, a2):
        ax.set_facecolor(BG)

    sc = a1.scatter(df["Glucose"], df["BMI"],
                    c=df["Outcome"], cmap="RdYlGn_r",
                    alpha=0.6, edgecolors="white", linewidths=0.3, s=40)
    plt.colorbar(sc, ax=a1, label="Diabetes (1=Yes)")
    a1.set(xlabel="Glucose (mg/dL)", ylabel="BMI",
           title="Glucose vs BMI -- Coloured by Outcome")
    a1.grid(alpha=0.25)
    a1.spines[["top","right"]].set_visible(False)

    counts = df["Outcome"].value_counts()
    a2.pie(counts.values, labels=["No Diabetes","Diabetes"],
           colors=[C_GREEN, C_RED], autopct="%1.1f%%", startangle=140,
           wedgeprops={"edgecolor":"white","linewidth":2})
    a2.set_title("Dataset Class Distribution")

    fig.tight_layout()
    _save(fig, "risk_distribution.png")


def plot_patient_radar(patient_norm, risk_score, rec_level, fuzzy_score,
                       feature_cols):
    labels = [c.replace("DiabetesPedigreeFunction","Pedigree")
               .replace("PhysicalActivity","Activity")
               .replace("BloodPressure","BP") for c in feature_cols]
    N    = len(labels)
    vals = list(patient_norm) + [patient_norm[0]]
    ang  = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True),
                           facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(ang, vals, color=C_BLUE, lw=2)
    ax.fill(ang, vals, color=C_BLUE, alpha=0.20)
    ax.set_thetagrids(np.degrees(ang[:-1]), labels, fontsize=9)
    cmap = {"STRICT": C_RED, "MODERATE": C_ORANGE, "MAINTAIN": C_GREEN}
    ax.set_title(
        f"Sample Patient Feature Profile\n"
        f"Risk: {risk_score*100:.1f}%  |  Rec: {rec_level}  "
        f"(FuzzyScore={fuzzy_score:.1f})",
        fontsize=11, fontweight="bold",
        color=cmap.get(rec_level, C_BLUE), pad=20
    )
    fig.tight_layout()
    _save(fig, "sample_patient_radar.png")

# ============================================================================
# MODULE F — SAMPLE PATIENT PREDICTION
# ============================================================================
def sample_prediction(model, scaler, sim, best_thresh, feature_cols):
    # [Preg, Glucose, BP, SkinThick, Insulin, BMI, Pedigree, Age,
    #  PhysActivity, DietScore, StressLevel]
    raw = np.array([[3, 148, 72, 35, 150, 33.6, 0.627, 50, 30, 25, 65]])
    scaled  = scaler.transform(raw)
    prob    = float(model.predict(scaled, verbose=0)[0][0])
    risk100 = prob * 100
    risk_label = ("High" if risk100 >= 60 else
                  "Medium" if risk100 >= 35 else "Low")

    activity_score = float(raw[0][8])
    sugar_score    = 100 - float(raw[0][9])

    rec_level, advice, fscore = fuzzy_recommend(sim, risk100,
                                                activity_score, sugar_score)

    print(f"\n{'='*65}")
    print("  SAMPLE PATIENT -- END-TO-END PREDICTION")
    print(f"{'='*65}")
    print(f"  ANN Diabetes Probability  : {prob*100:.2f} %")
    print(f"  Diabetes Risk Level       : {risk_label}  ({risk100:.1f} / 100)")
    print(f"  Fuzzy Rec. Score          : {fscore:.1f} / 100  ->  {rec_level}")
    print(f"\n  Personalised Recommendations:")
    for line in advice:
        print(line)

    norm = (raw[0] - raw[0].min()) / (raw[0].max() - raw[0].min() + 1e-8)
    plot_patient_radar(norm, prob, rec_level, fscore, feature_cols)

# ============================================================================
# MAIN
# ============================================================================
def main():
    X_train, X_test, y_train, y_test, scaler, df, feature_cols = \
        load_and_preprocess(DATASET_PATH)

    model, y_pred, y_prob, histories, best_thresh = \
        train_ann(X_train, y_train, X_test, y_test)

    acc, prec, rec, f1, rocauc, fpr, tpr, cm = \
        evaluate(y_test, y_pred, y_prob)

    print_knowledge_base()
    sim, risk_v, act_v, sug_v, rec_v = build_fuzzy_system()
    print("[FUZZY] Control system built -- 27 rules active.")

    print(f"\n[GRAPHS] Saving all plots to ./{GRAPHS_DIR}/")
    plot_confusion_matrix(cm)
    plot_roc(fpr, tpr, rocauc)
    plot_metrics_bar(acc, prec, rec, f1, rocauc)
    plot_training_history(histories)
    plot_feature_importance(X_train, y_train, feature_cols)
    plot_fuzzy_mf(risk_v, act_v, sug_v, rec_v)
    plot_risk_distribution(df)

    sample_prediction(model, scaler, sim, best_thresh, feature_cols)

    print(f"\n{'='*65}")
    print("  SYSTEM COMPLETE")
    print(f"{'='*65}")
    print(f"  Graphs saved in : ./{GRAPHS_DIR}/")
    print(f"  Test Accuracy   : {acc*100:.2f} %\n")


if __name__ == "__main__":
    main()