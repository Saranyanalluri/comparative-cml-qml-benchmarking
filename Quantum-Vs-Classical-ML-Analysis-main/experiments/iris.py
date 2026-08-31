import time
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute


# ============================================================
# IRIS 8-MODEL CML vs QML BENCHMARK
#
# CLASSICAL
#   1. SVM
#   2. Random Forest
#   3. XGBoost
#   4. LightGBM
#
# QUANTUM / HYBRID
#   5. QSVC
#   6. Quantum-Enhanced Random Forest (QERF)
#   7. Quantum-Enhanced XGBoost (QEXGB)
#   8. Quantum-Enhanced LightGBM (QELGBM)
#
# Quantum representation:
#   MinMaxScaler [0, pi]
#   ZZFeatureMap
#   reps = 1
#   entanglement = linear
#
# Evaluation:
#   5-fold Stratified Cross Validation
#   Accuracy, Precision, Recall, F1, Runtime
# ============================================================


print("=" * 75)
print("IRIS DATASET - 8 MODEL CML vs QML BENCHMARK")
print("=" * 75)


# ============================================================
# 1. LOAD FULL IRIS DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

N_FEATURES = X.shape[1]
N_CLASSES = len(np.unique(y))

print("\nDataset Information")
print("-" * 50)
print(f"Samples       : {X.shape[0]}")
print(f"Features      : {N_FEATURES}")
print(f"Classes       : {N_CLASSES}")
print(f"Class names   : {list(iris.target_names)}")


# ============================================================
# 2. CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 3. RESULT STORAGE
# ============================================================

results = []


# ============================================================
# 4. QUANTUM FEATURE MAP
#
# This configuration was selected from the previous Iris
# diagnostic experiment:
#
# MinMaxScaler [0, pi]
# ZZFeatureMap
# reps = 1
# linear entanglement
# ============================================================

feature_map = ZZFeatureMap(
    feature_dimension=N_FEATURES,
    reps=1,
    entanglement="linear"
)


# ============================================================
# 5. QUANTUM SAMPLER AND FIDELITY
# ============================================================

sampler = StatevectorSampler()

fidelity = ComputeUncompute(
    sampler=sampler
)


# ============================================================
# 6. QUANTUM KERNEL
# ============================================================

quantum_kernel = FidelityQuantumKernel(
    fidelity=fidelity,
    feature_map=feature_map
)


# ============================================================
# 7. 5-FOLD CROSS VALIDATION
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    skf.split(X, y),
    1
):

    print("\n" + "=" * 75)
    print(f"FOLD {fold}")
    print("=" * 75)


    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]


    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )


    # ========================================================
    # PREPROCESSING
    #
    # Fit scaler ONLY on training data.
    #
    # Quantum feature maps interpret input values as angles,
    # therefore values are mapped to [0, pi].
    # ========================================================

    scaler = MinMaxScaler(
        feature_range=(0, np.pi)
    )

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # ========================================================
    # HELPER FUNCTION FOR METRICS
    # ========================================================

    def calculate_metrics(y_true, y_pred):

        return (
            accuracy_score(y_true, y_pred),

            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        )


    # ========================================================
    # 1. SVM
    # ========================================================

    print("\n[1/8] Running SVM...")

    svm = SVC(
        kernel="rbf"
    )

    start = time.time()

    svm.fit(
        X_train_scaled,
        y_train
    )

    svm_pred = svm.predict(
        X_test_scaled
    )

    svm_time = time.time() - start

    (
        svm_accuracy,
        svm_precision,
        svm_recall,
        svm_f1
    ) = calculate_metrics(
        y_test,
        svm_pred
    )

    print(f"SVM Accuracy : {svm_accuracy:.4f}")
    print(f"SVM F1       : {svm_f1:.4f}")
    print(f"SVM Time     : {svm_time:.4f} sec")


    # ========================================================
    # 2. RANDOM FOREST
    # ========================================================

    print("\n[2/8] Running Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    start = time.time()

    rf.fit(
        X_train_scaled,
        y_train
    )

    rf_pred = rf.predict(
        X_test_scaled
    )

    rf_time = time.time() - start

    (
        rf_accuracy,
        rf_precision,
        rf_recall,
        rf_f1
    ) = calculate_metrics(
        y_test,
        rf_pred
    )

    print(f"RF Accuracy : {rf_accuracy:.4f}")
    print(f"RF F1       : {rf_f1:.4f}")
    print(f"RF Time     : {rf_time:.4f} sec")


    # ========================================================
    # 3. XGBOOST
    # ========================================================

    print("\n[3/8] Running XGBoost...")

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    start = time.time()

    xgb.fit(
        X_train_scaled,
        y_train
    )

    xgb_pred = xgb.predict(
        X_test_scaled
    )

    xgb_time = time.time() - start

    (
        xgb_accuracy,
        xgb_precision,
        xgb_recall,
        xgb_f1
    ) = calculate_metrics(
        y_test,
        xgb_pred
    )

    print(f"XGBoost Accuracy : {xgb_accuracy:.4f}")
    print(f"XGBoost F1       : {xgb_f1:.4f}")
    print(f"XGBoost Time     : {xgb_time:.4f} sec")


    # ========================================================
    # 4. LIGHTGBM
    # ========================================================

    print("\n[4/8] Running LightGBM...")

    lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        objective="multiclass",
        num_class=N_CLASSES,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    start = time.time()

    lgbm.fit(
        X_train_scaled,
        y_train
    )

    lgbm_pred = lgbm.predict(
        X_test_scaled
    )

    lgbm_time = time.time() - start

    (
        lgbm_accuracy,
        lgbm_precision,
        lgbm_recall,
        lgbm_f1
    ) = calculate_metrics(
        y_test,
        lgbm_pred
    )

    print(f"LightGBM Accuracy : {lgbm_accuracy:.4f}")
    print(f"LightGBM F1       : {lgbm_f1:.4f}")
    print(f"LightGBM Time     : {lgbm_time:.4f} sec")


    print("\n[5/8] Running QSVC...")

    qsvc = QSVC(
        quantum_kernel=quantum_kernel
    )

    start = time.time()

    qsvc.fit(
        X_train_scaled,
        y_train
    )

    qsvc_pred = qsvc.predict(
        X_test_scaled
    )

    qsvc_time = time.time() - start

    (
        qsvc_accuracy,
        qsvc_precision,
        qsvc_recall,
        qsvc_f1
    ) = calculate_metrics(
        y_test,
        qsvc_pred
    )

    print(f"QSVC Accuracy : {qsvc_accuracy:.4f}")
    print(f"QSVC F1       : {qsvc_f1:.4f}")
    print(f"QSVC Time     : {qsvc_time:.4f} sec")


    # ========================================================
    # QUANTUM KERNEL REPRESENTATION
    #
    # Used by:
    #   QERF
    #   QEXGB
    #   QELGBM
    #
    # The kernel is calculated once per fold and reused.
    # ========================================================

    print("\nCalculating quantum-kernel representation...")

    quantum_start = time.time()

    K_train = quantum_kernel.evaluate(
        x_vec=X_train_scaled
    )

    K_test = quantum_kernel.evaluate(
        x_vec=X_test_scaled,
        y_vec=X_train_scaled
    )

    quantum_feature_time = time.time() - quantum_start

    print(
        f"Quantum representation time : "
        f"{quantum_feature_time:.2f} sec"
    )

    print(
        f"Quantum train representation : "
        f"{K_train.shape}"
    )

    print(
        f"Quantum test representation  : "
        f"{K_test.shape}"
    )


    # ========================================================
    # 6. QUANTUM-ENHANCED RANDOM FOREST
    #
    # Quantum kernel -> Random Forest
    # ========================================================

    print("\n[6/8] Running Quantum-Enhanced Random Forest...")

    qerf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    qerf_start = time.time()

    qerf.fit(
        K_train,
        y_train
    )

    qerf_pred = qerf.predict(
        K_test
    )

    qerf_model_time = time.time() - qerf_start

    qerf_time = (
        quantum_feature_time +
        qerf_model_time
    )

    (
        qerf_accuracy,
        qerf_precision,
        qerf_recall,
        qerf_f1
    ) = calculate_metrics(
        y_test,
        qerf_pred
    )

    print(f"QERF Accuracy : {qerf_accuracy:.4f}")
    print(f"QERF F1       : {qerf_f1:.4f}")
    print(f"QERF Time     : {qerf_time:.4f} sec")


    print("\n[7/8] Running Quantum-Enhanced XGBoost...")

    qexgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    qexgb_start = time.time()

    qexgb.fit(
        K_train,
        y_train
    )

    qexgb_pred = qexgb.predict(
        K_test
    )

    qexgb_model_time = time.time() - qexgb_start

    qexgb_time = (
        quantum_feature_time +
        qexgb_model_time
    )

    (
        qexgb_accuracy,
        qexgb_precision,
        qexgb_recall,
        qexgb_f1
    ) = calculate_metrics(
        y_test,
        qexgb_pred
    )

    print(f"QEXGB Accuracy : {qexgb_accuracy:.4f}")
    print(f"QEXGB F1       : {qexgb_f1:.4f}")
    print(f"QEXGB Time     : {qexgb_time:.4f} sec")


    print("\n[8/8] Running Quantum-Enhanced LightGBM...")

    qelgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        objective="multiclass",
        num_class=N_CLASSES,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    qelgbm_start = time.time()

    qelgbm.fit(
        K_train,
        y_train
    )

    qelgbm_pred = qelgbm.predict(
        K_test
    )

    qelgbm_model_time = time.time() - qelgbm_start

    qelgbm_time = (
        quantum_feature_time +
        qelgbm_model_time
    )

    (
        qelgbm_accuracy,
        qelgbm_precision,
        qelgbm_recall,
        qelgbm_f1
    ) = calculate_metrics(
        y_test,
        qelgbm_pred
    )

    print(f"QELGBM Accuracy : {qelgbm_accuracy:.4f}")
    print(f"QELGBM F1       : {qelgbm_f1:.4f}")
    print(f"QELGBM Time     : {qelgbm_time:.4f} sec")


    results.append({

        "Fold": fold,

        # ---------------- SVM ----------------
        "SVM_Accuracy": svm_accuracy,
        "SVM_Precision": svm_precision,
        "SVM_Recall": svm_recall,
        "SVM_F1": svm_f1,
        "SVM_Time": svm_time,

        # ---------------- QSVC ----------------
        "QSVC_Accuracy": qsvc_accuracy,
        "QSVC_Precision": qsvc_precision,
        "QSVC_Recall": qsvc_recall,
        "QSVC_F1": qsvc_f1,
        "QSVC_Time": qsvc_time,

        # ---------------- RF ----------------
        "RF_Accuracy": rf_accuracy,
        "RF_Precision": rf_precision,
        "RF_Recall": rf_recall,
        "RF_F1": rf_f1,
        "RF_Time": rf_time,

        # ---------------- QERF ----------------
        "QERF_Accuracy": qerf_accuracy,
        "QERF_Precision": qerf_precision,
        "QERF_Recall": qerf_recall,
        "QERF_F1": qerf_f1,
        "QERF_Time": qerf_time,

        # ---------------- XGBoost ----------------
        "XGBoost_Accuracy": xgb_accuracy,
        "XGBoost_Precision": xgb_precision,
        "XGBoost_Recall": xgb_recall,
        "XGBoost_F1": xgb_f1,
        "XGBoost_Time": xgb_time,

        # ---------------- QEXGB ----------------
        "QEXGB_Accuracy": qexgb_accuracy,
        "QEXGB_Precision": qexgb_precision,
        "QEXGB_Recall": qexgb_recall,
        "QEXGB_F1": qexgb_f1,
        "QEXGB_Time": qexgb_time,

        # ---------------- LightGBM ----------------
        "LightGBM_Accuracy": lgbm_accuracy,
        "LightGBM_Precision": lgbm_precision,
        "LightGBM_Recall": lgbm_recall,
        "LightGBM_F1": lgbm_f1,
        "LightGBM_Time": lgbm_time,

        # ---------------- QELGBM ----------------
        "QELGBM_Accuracy": qelgbm_accuracy,
        "QELGBM_Precision": qelgbm_precision,
        "QELGBM_Recall": qelgbm_recall,
        "QELGBM_F1": qelgbm_f1,
        "QELGBM_Time": qelgbm_time
    })


    print("\n" + "-" * 75)
    print(f"Fold {fold} completed.")
    print("-" * 75)


results_df = pd.DataFrame(
    results
)

results_file = "iris_8model_benchmark_results.csv"

results_df.to_csv(
    results_file,
    index=False
)

print("\n" + "=" * 75)
print("FOLD RESULTS SAVED")
print("=" * 75)

print(
    f"File: {results_file}"
)

models = [

    (
        "SVM",
        "SVM_Accuracy",
        "SVM_Precision",
        "SVM_Recall",
        "SVM_F1",
        "SVM_Time"
    ),

    (
        "QSVC",
        "QSVC_Accuracy",
        "QSVC_Precision",
        "QSVC_Recall",
        "QSVC_F1",
        "QSVC_Time"
    ),

    (
        "Random Forest",
        "RF_Accuracy",
        "RF_Precision",
        "RF_Recall",
        "RF_F1",
        "RF_Time"
    ),

    (
        "Quantum-Enhanced RF",
        "QERF_Accuracy",
        "QERF_Precision",
        "QERF_Recall",
        "QERF_F1",
        "QERF_Time"
    ),

    (
        "XGBoost",
        "XGBoost_Accuracy",
        "XGBoost_Precision",
        "XGBoost_Recall",
        "XGBoost_F1",
        "XGBoost_Time"
    ),

    (
        "Quantum-Enhanced XGBoost",
        "QEXGB_Accuracy",
        "QEXGB_Precision",
        "QEXGB_Recall",
        "QEXGB_F1",
        "QEXGB_Time"
    ),

    (
        "LightGBM",
        "LightGBM_Accuracy",
        "LightGBM_Precision",
        "LightGBM_Recall",
        "LightGBM_F1",
        "LightGBM_Time"
    ),

    (
        "Quantum-Enhanced LightGBM",
        "QELGBM_Accuracy",
        "QELGBM_Precision",
        "QELGBM_Recall",
        "QELGBM_F1",
        "QELGBM_Time"
    )
]


summary_rows = []


for (
    model_name,
    accuracy_col,
    precision_col,
    recall_col,
    f1_col,
    time_col
) in models:

    summary_rows.append({

        "Model": model_name,

        "Average_Accuracy":
            results_df[accuracy_col].mean(),

        "Std_Accuracy":
            results_df[accuracy_col].std(),

        "Average_Precision":
            results_df[precision_col].mean(),

        "Average_Recall":
            results_df[recall_col].mean(),

        "Average_F1":
            results_df[f1_col].mean(),

        "Average_Runtime_sec":
            results_df[time_col].mean()
    })


summary_df = pd.DataFrame(
    summary_rows
)

print("\n" + "=" * 75)
print("FINAL IRIS 8-MODEL BENCHMARK SUMMARY")
print("=" * 75)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

summary_file = "iris_8model_benchmark_summary.csv"

summary_df.to_csv(
    summary_file,
    index=False
)

print("\n" + "=" * 75)
print("SUMMARY SAVED")
print("=" * 75)

print(
    f"File: {summary_file}"
)

print("\nBenchmark completed successfully.")