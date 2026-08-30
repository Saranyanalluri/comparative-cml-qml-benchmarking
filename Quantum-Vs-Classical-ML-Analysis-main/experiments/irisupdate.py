import numpy as np
import time
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

print("Loading Iris Dataset...")

iris = load_iris()
X = iris.data
y = iris.target

# Binary Classification
mask = y != 2
X = X[mask]
y = y[mask]

# Use first 2 features for quantum models
n_qubits = 2
X = X[:, :n_qubits]

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

lr_scores = []
knn_scores = []
rf_scores = []
svm_scores = []
qsvc_scores = []
pegasos_scores = []

lr_times = []
knn_times = []
rf_times = []
svm_times = []
qsvc_times = []
pegasos_times = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

    print(f"\n===== Fold {fold + 1} =====")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =====================================
    # Logistic Regression
    # =====================================

    lr_model = LogisticRegression(max_iter=1000)

    start = time.time()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_time = time.time() - start

    lr_acc = accuracy_score(y_test, lr_pred)

    lr_scores.append(lr_acc)
    lr_times.append(lr_time)

    print(f"LR Accuracy: {lr_acc:.4f}")

    # =====================================
    # KNN
    # =====================================

    knn_model = KNeighborsClassifier(n_neighbors=5)

    start = time.time()
    knn_model.fit(X_train_scaled, y_train)
    knn_pred = knn_model.predict(X_test_scaled)
    knn_time = time.time() - start

    knn_acc = accuracy_score(y_test, knn_pred)

    knn_scores.append(knn_acc)
    knn_times.append(knn_time)

    print(f"KNN Accuracy: {knn_acc:.4f}")

    # =====================================
    # Random Forest
    # =====================================

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    start = time.time()
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_time = time.time() - start

    rf_acc = accuracy_score(y_test, rf_pred)

    rf_scores.append(rf_acc)
    rf_times.append(rf_time)

    print(f"RF Accuracy: {rf_acc:.4f}")

    # =====================================
    # SVM
    # =====================================

    svm_model = SVC(kernel="rbf")

    start = time.time()
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_time = time.time() - start

    svm_acc = accuracy_score(y_test, svm_pred)

    svm_scores.append(svm_acc)
    svm_times.append(svm_time)

    print(f"SVM Accuracy: {svm_acc:.4f}")

    # =====================================
    # Quantum Kernel Setup
    # =====================================

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

    # =====================================
    # QSVC
    # =====================================

    qsvc_model = QSVC(
        quantum_kernel=quantum_kernel
    )

    start = time.time()
    qsvc_model.fit(X_train_scaled, y_train)
    qsvc_pred = qsvc_model.predict(X_test_scaled)
    qsvc_time = time.time() - start

    qsvc_acc = accuracy_score(y_test, qsvc_pred)

    qsvc_scores.append(qsvc_acc)
    qsvc_times.append(qsvc_time)

    print(f"QSVC Accuracy: {qsvc_acc:.4f}")

    # =====================================
    # PegasosQSVC
    # =====================================

    pegasos_model = PegasosQSVC(
        quantum_kernel=quantum_kernel
    )

    start = time.time()
    pegasos_model.fit(X_train_scaled, y_train)
    pegasos_pred = pegasos_model.predict(X_test_scaled)
    pegasos_time = time.time() - start

    pegasos_acc = accuracy_score(
        y_test,
        pegasos_pred
    )

    pegasos_scores.append(pegasos_acc)
    pegasos_times.append(pegasos_time)

    print(f"PegasosQSVC Accuracy: {pegasos_acc:.4f}")

# =====================================
# Final Results
# =====================================

print("\n========== FINAL RESULTS ==========")

print(f"LR Accuracy:       {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")
print(f"KNN Accuracy:      {np.mean(knn_scores):.4f} ± {np.std(knn_scores):.4f}")
print(f"RF Accuracy:       {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
print(f"SVM Accuracy:      {np.mean(svm_scores):.4f} ± {np.std(svm_scores):.4f}")
print(f"QSVC Accuracy:     {np.mean(qsvc_scores):.4f} ± {np.std(qsvc_scores):.4f}")
print(f"Pegasos Accuracy:  {np.mean(pegasos_scores):.4f} ± {np.std(pegasos_scores):.4f}")

print("\n===== Average Runtime =====")

print(f"LR Time:       {np.mean(lr_times):.4f}s")
print(f"KNN Time:      {np.mean(knn_times):.4f}s")
print(f"RF Time:       {np.mean(rf_times):.4f}s")
print(f"SVM Time:      {np.mean(svm_times):.4f}s")
print(f"QSVC Time:     {np.mean(qsvc_times):.4f}s")
print(f"Pegasos Time:  {np.mean(pegasos_times):.4f}s")

# =====================================
# Save Results
# =====================================

results_df = pd.DataFrame({
    "LR": lr_scores,
    "KNN": knn_scores,
    "RF": rf_scores,
    "SVM": svm_scores,
    "QSVC": qsvc_scores,
    "PegasosQSVC": pegasos_scores,
    "LR_Time": lr_times,
    "KNN_Time": knn_times,
    "RF_Time": rf_times,
    "SVM_Time": svm_times,
    "QSVC_Time": qsvc_times,
    "Pegasos_Time": pegasos_times
})

results_df.to_csv(
    "iris_multimodel_results.csv",
    index=False
)

print("\nResults saved to iris_multimodel_results.csv")