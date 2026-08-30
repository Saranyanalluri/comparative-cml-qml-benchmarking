import numpy as np
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute


# ============================================================
# ISING DATASET
# SVM vs QSVC - 5 Fold Cross Validation
# ============================================================

print("=" * 60)
print("Ising Dataset - SVM vs QSVC Benchmark")
print("=" * 60)

np.random.seed(42)
# ============================================================
# 1. Generate Ising Dataset
# ============================================================

def generate_ising_data(n_samples, n_spins):

    X = []
    y = []

    for _ in range(n_samples):

        # 0 = Ordered / Ferromagnetic
        # 1 = Disordered / Paramagnetic
        label = np.random.randint(2)

        y.append(label)

        if label == 0:

            # Ordered phase
            base_state = np.random.randint(2)

            spins = [
                base_state
                if np.random.rand() > 0.1
                else 1 - base_state
                for _ in range(n_spins)
            ]

        else:

            # Disordered phase
            spins = np.random.randint(
                2,
                size=n_spins
            )

        X.append(spins)

    return np.array(X), np.array(y)


# ============================================================
# 2. Dataset Parameters
# ============================================================

n_qubits = 9
n_samples = 200

print("\nGenerating Ising dataset...")

X, y = generate_ising_data(
    n_samples,
    n_qubits
)

print("Total Samples     :", len(X))
print("Feature Dimension :", X.shape[1])
print("Number of Classes :", len(np.unique(y)))


# ============================================================
# 3. 5-Fold Stratified Cross Validation
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# Result Lists
# ============================================================

svm_scores = []
qsvc_scores = []

svm_times = []
qsvc_times = []


print("\nStarting 5-Fold Cross Validation...")


# ============================================================
# 4. Cross Validation
# ============================================================

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
    # Classical SVM
    # ========================================================

    svm = SVC(
        kernel="rbf"
    )

    start_time = time.time()

    svm.fit(
        X_train,
        y_train
    )

    svm_pred = svm.predict(
        X_test
    )

    svm_time = time.time() - start_time

    svm_accuracy = accuracy_score(
        y_test,
        svm_pred
    )

    svm_scores.append(
        svm_accuracy
    )

    svm_times.append(
        svm_time
    )

    print(
        f"SVM Accuracy  : "
        f"{svm_accuracy:.4f}"
    )

    print(
        f"SVM Time      : "
        f"{svm_time:.4f} sec"
    )


    # ========================================================
    # Quantum Kernel
    # ========================================================

    sampler = Sampler()

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


    # ========================================================
    # QSVC
    # ========================================================

    qsvc = QSVC(
        quantum_kernel=quantum_kernel
    )

    start_time = time.time()

    qsvc.fit(
        X_train,
        y_train
    )

    qsvc_pred = qsvc.predict(
        X_test
    )

    qsvc_time = time.time() - start_time

    qsvc_accuracy = accuracy_score(
        y_test,
        qsvc_pred
    )

    qsvc_scores.append(
        qsvc_accuracy
    )

    qsvc_times.append(
        qsvc_time
    )

    print(
        f"QSVC Accuracy : "
        f"{qsvc_accuracy:.4f}"
    )

    print(
        f"QSVC Time     : "
        f"{qsvc_time:.4f} sec"
    )


# ============================================================
# 5. Fold-Wise Results
# ============================================================

results_df = pd.DataFrame({

    "Fold": range(1, 6),

    "CML_Score": svm_scores,

    "QML_Score": qsvc_scores,

    "CML_Time": svm_times,

    "QML_Time": qsvc_times

})


print("\n" + "=" * 60)
print("FOLD-WISE RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))


# ============================================================
# 6. Save Results
# ============================================================

results_df.to_csv(
    "ising_cv_results.csv",
    index=False
)

print(
    "\nResults saved to "
    "ising_cv_results.csv"
)


# ============================================================
# 7. Final Performance
# ============================================================

avg_svm_accuracy = np.mean(
    svm_scores
)

avg_qsvc_accuracy = np.mean(
    qsvc_scores
)

avg_svm_time = np.mean(
    svm_times
)

avg_qsvc_time = np.mean(
    qsvc_times
)


print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(
    f"SVM Average Accuracy  : "
    f"{avg_svm_accuracy:.4f}"
)

print(
    f"QSVC Average Accuracy : "
    f"{avg_qsvc_accuracy:.4f}"
)

print(
    f"SVM Average Time      : "
    f"{avg_svm_time:.4f} sec"
)

print(
    f"QSVC Average Time     : "
    f"{avg_qsvc_time:.4f} sec"
)


# ============================================================
# 8. Summary CSV
# ============================================================

summary_df = pd.DataFrame({

    "Model": [
        "SVM",
        "QSVC"
    ],

    "Average Accuracy": [
        avg_svm_accuracy,
        avg_qsvc_accuracy
    ],

    "Average Time (s)": [
        avg_svm_time,
        avg_qsvc_time
    ]

})


summary_df.to_csv(
    "ising_summary.csv",
    index=False
)


print(
    "\nSummary saved to "
    "ising_summary.csv"
)

print("\nBenchmark Completed Successfully.")