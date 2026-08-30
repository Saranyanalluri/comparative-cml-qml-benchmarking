import numpy as np
import pandas as pd
import time

from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute


# ============================================================
# RadioML / Dataset Style Benchmark
# Moons Multi-Model Benchmark
# ============================================================

print("=" * 60)
print("Moons Multi-Model Benchmark")
print("=" * 60)


# ============================================================
# 1. Generate Moons Dataset
# ============================================================

n_samples = 200

X, y = make_moons(
    n_samples=n_samples,
    noise=0.15,
    random_state=42
)

print("\nDataset Loaded Successfully")

print("Total Samples :", len(X))
print("Feature Dimension :", X.shape[1])
print("Classes :", len(np.unique(y)))


# ============================================================
# 2. Feature Scaling
# ============================================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

print("Feature Scaling Completed")


# ============================================================
# 3. Quantum Feature Map
# ============================================================

n_qubits = X.shape[1]

sampler = StatevectorSampler()

fidelity = ComputeUncompute(
    sampler=sampler
)

feature_map = ZZFeatureMap(
    feature_dimension=n_qubits,
    reps=2,
    entanglement="linear"
)

quantum_kernel = FidelityQuantumKernel(
    fidelity=fidelity,
    feature_map=feature_map
)


# ============================================================
# 4. 5-Fold Stratified Cross Validation
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. Result Lists
# ============================================================

lr_scores = []
knn_scores = []
rf_scores = []
svm_scores = []
qsvc_scores = []

lr_times = []
knn_times = []
rf_times = []
svm_times = []
qsvc_times = []


# ============================================================
# 6. Cross Validation
# ============================================================

print("\nStarting 5-Fold Cross Validation...")


for fold, (train_idx, test_idx) in enumerate(
    skf.split(X, y),
    1
):

    print("\n" + "=" * 45)
    print(f"Fold {fold}")
    print("=" * 45)


    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]


    # ========================================================
    # Logistic Regression
    # ========================================================

    model = LogisticRegression(
        max_iter=1000
    )

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    elapsed = time.time() - start

    accuracy = accuracy_score(
        y_test,
        pred
    )

    lr_scores.append(accuracy)
    lr_times.append(elapsed)

    print(
        f"LR Accuracy   : {accuracy:.4f}"
    )

    print(
        f"LR Time       : {elapsed:.4f} sec"
    )


    # ========================================================
    # KNN
    # ========================================================

    model = KNeighborsClassifier(
        n_neighbors=5
    )

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    elapsed = time.time() - start

    accuracy = accuracy_score(
        y_test,
        pred
    )

    knn_scores.append(accuracy)
    knn_times.append(elapsed)

    print(
        f"KNN Accuracy  : {accuracy:.4f}"
    )

    print(
        f"KNN Time      : {elapsed:.4f} sec"
    )


    # ========================================================
    # Random Forest
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    elapsed = time.time() - start

    accuracy = accuracy_score(
        y_test,
        pred
    )

    rf_scores.append(accuracy)
    rf_times.append(elapsed)

    print(
        f"RF Accuracy   : {accuracy:.4f}"
    )

    print(
        f"RF Time       : {elapsed:.4f} sec"
    )


    # ========================================================
    # Support Vector Machine
    # ========================================================

    model = SVC(
        kernel="rbf"
    )

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    elapsed = time.time() - start

    accuracy = accuracy_score(
        y_test,
        pred
    )

    svm_scores.append(accuracy)
    svm_times.append(elapsed)

    print(
        f"SVM Accuracy  : {accuracy:.4f}"
    )

    print(
        f"SVM Time      : {elapsed:.4f} sec"
    )


    # ========================================================
    # Quantum Support Vector Classifier
    # ========================================================

    qsvc = QSVC(
        quantum_kernel=quantum_kernel
    )

    start = time.time()

    qsvc.fit(
        X_train,
        y_train
    )

    pred = qsvc.predict(
        X_test
    )

    elapsed = time.time() - start

    accuracy = accuracy_score(
        y_test,
        pred
    )

    qsvc_scores.append(accuracy)
    qsvc_times.append(elapsed)

    print(
        f"QSVC Accuracy : {accuracy:.4f}"
    )

    print(
        f"QSVC Time     : {elapsed:.4f} sec"
    )


# ============================================================
# 7. Final Results
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Logistic Regression : "
    f"{np.mean(lr_scores):.4f} ± "
    f"{np.std(lr_scores):.4f}"
)

print(
    f"KNN                 : "
    f"{np.mean(knn_scores):.4f} ± "
    f"{np.std(knn_scores):.4f}"
)

print(
    f"Random Forest       : "
    f"{np.mean(rf_scores):.4f} ± "
    f"{np.std(rf_scores):.4f}"
)

print(
    f"SVM                 : "
    f"{np.mean(svm_scores):.4f} ± "
    f"{np.std(svm_scores):.4f}"
)

print(
    f"QSVC                : "
    f"{np.mean(qsvc_scores):.4f} ± "
    f"{np.std(qsvc_scores):.4f}"
)


# ============================================================
# 8. Average Runtime
# ============================================================

print("\n" + "=" * 60)
print("AVERAGE RUNTIME")
print("=" * 60)

print(
    f"Logistic Regression : "
    f"{np.mean(lr_times):.4f} sec"
)

print(
    f"KNN                 : "
    f"{np.mean(knn_times):.4f} sec"
)

print(
    f"Random Forest       : "
    f"{np.mean(rf_times):.4f} sec"
)

print(
    f"SVM                 : "
    f"{np.mean(svm_times):.4f} sec"
)

print(
    f"QSVC                : "
    f"{np.mean(qsvc_times):.4f} sec"
)


# ============================================================
# 9. Fold-Wise Results
# ============================================================

results_df = pd.DataFrame({

    "Fold": range(1, 6),

    "LR": lr_scores,
    "KNN": knn_scores,
    "RF": rf_scores,
    "SVM": svm_scores,
    "QSVC": qsvc_scores,

    "LR_Time": lr_times,
    "KNN_Time": knn_times,
    "RF_Time": rf_times,
    "SVM_Time": svm_times,
    "QSVC_Time": qsvc_times

})


# ============================================================
# 10. Save Results
# ============================================================

results_df.to_csv(
    "moons_multimodel_results.csv",
    index=False
)

print(
    "\nResults saved to "
    "moons_multimodel_results.csv"
)


# ============================================================
# 11. Summary
# ============================================================

summary_df = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "KNN",
        "Random Forest",
        "SVM",
        "QSVC"
    ],

    "Average Accuracy": [

        np.mean(lr_scores),
        np.mean(knn_scores),
        np.mean(rf_scores),
        np.mean(svm_scores),
        np.mean(qsvc_scores)

    ],

    "Average Time (s)": [

        np.mean(lr_times),
        np.mean(knn_times),
        np.mean(rf_times),
        np.mean(svm_times),
        np.mean(qsvc_times)

    ]

})


summary_df.to_csv(
    "moons_multimodel_summary.csv",
    index=False
)


# ============================================================
# 12. Display Summary
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    summary_df.to_string(index=False)
)

print(
    "\nSummary saved to "
    "moons_multimodel_summary.csv"
)

print("\nBenchmark Completed Successfully.")