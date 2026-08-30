import numpy as np
import pandas as pd
import time

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

def generate_parity_data(n_samples, n_features):

    X = np.random.randint(
        2,
        size=(n_samples, n_features)
    )

    y = np.sum(
        X,
        axis=1
    ) % 2

    return X, y


print("Loading Bit Parity Dataset...")

n_qubits = 6
n_samples = 100

X, y = generate_parity_data(
    n_samples,
    n_qubits
)

print("Dataset Loaded")
print("Samples :", len(X))


sampler = Sampler()

fidelity = ComputeUncompute(
    sampler=sampler
)

feature_map = ZZFeatureMap(
    feature_dimension=n_qubits,
    reps=2,
    entanglement="full"
)

quantum_kernel = FidelityQuantumKernel(
    fidelity=fidelity,
    feature_map=feature_map
)


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

fold = 1

print("\nStarting 5-Fold Cross Validation...\n")
# =====================================================
# 5-Fold Cross Validation
# =====================================================

for train_idx, test_idx in skf.split(X, y):

    print(f"\n========== Fold {fold} ==========")

    X_train_fold = X[train_idx]
    X_test_fold = X[test_idx]

    y_train_fold = y[train_idx]
    y_test_fold = y[test_idx]

    # -----------------------------
    # Classical SVM
    # -----------------------------

    svm = SVC(kernel="rbf")

    start = time.time()

    svm.fit(X_train_fold, y_train_fold)

    svm_pred = svm.predict(X_test_fold)

    svm_time = time.time() - start

    svm_score = accuracy_score(
        y_test_fold,
        svm_pred
    )

    print(f"SVM Accuracy : {svm_score:.4f}")
    print(f"SVM Time     : {svm_time:.4f} sec")

    # -----------------------------
    # Quantum SVM
    # -----------------------------

    qsvc = QSVC(
        quantum_kernel=quantum_kernel
    )

    start = time.time()

    qsvc.fit(
        X_train_fold,
        y_train_fold
    )

    qsvc_pred = qsvc.predict(
        X_test_fold
    )

    qsvc_time = time.time() - start

    qsvc_score = accuracy_score(
        y_test_fold,
        qsvc_pred
    )

    print(f"QSVC Accuracy: {qsvc_score:.4f}")
    print(f"QSVC Time    : {qsvc_time:.4f} sec")

    results.append({
        "Fold": fold,
        "CML_Score": svm_score,
        "QML_Score": qsvc_score,
        "CML_Time": svm_time,
        "QML_Time": qsvc_time
    })

    fold += 1

# =====================================================
# Save Results
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "bit_parity_results.csv",
    index=False
)

print("\nResults saved as bit_parity_results.csv")


avg_cml_score = results_df["CML_Score"].mean()
avg_qml_score = results_df["QML_Score"].mean()

avg_cml_time = results_df["CML_Time"].mean()
avg_qml_time = results_df["QML_Time"].mean()

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print(f"Average CML Accuracy : {avg_cml_score:.4f}")
print(f"Average QML Accuracy : {avg_qml_score:.4f}")

print(f"Average CML Time     : {avg_cml_time:.4f} sec")
print(f"Average QML Time     : {avg_qml_time:.4f} sec")

summary = pd.DataFrame({
    "Metric": [
        "Average CML Accuracy",
        "Average QML Accuracy",
        "Average CML Time",
        "Average QML Time"
    ],
    "Value": [
        avg_cml_score,
        avg_qml_score,
        avg_cml_time,
        avg_qml_time
    ]
})

summary.to_csv(
    "bit_parity_summary.csv",
    index=False
)

print("\nSummary saved as bars_stripes_summary.csv")