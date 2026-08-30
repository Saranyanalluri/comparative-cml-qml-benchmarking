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

def generate_bars_and_stripes(n_samples, n_features_sqrt):

    n_features = n_features_sqrt * n_features_sqrt

    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    for i in range(n_samples):

        label = np.random.randint(2)

        y[i] = label

        img = np.zeros((n_features_sqrt, n_features_sqrt))

        if label == 0:

            col_pattern = np.random.randint(
                2,
                size=n_features_sqrt
            )

            if np.sum(col_pattern) == 0:
                col_pattern[np.random.randint(n_features_sqrt)] = 1

            for col in range(n_features_sqrt):
                if col_pattern[col]:
                    img[:, col] = 1

        else:

            row_pattern = np.random.randint(
                2,
                size=n_features_sqrt
            )

            if np.sum(row_pattern) == 0:
                row_pattern[np.random.randint(n_features_sqrt)] = 1

            for row in range(n_features_sqrt):
                if row_pattern[row]:
                    img[row, :] = 1

        noise = np.random.rand(
            n_features_sqrt,
            n_features_sqrt
        ) < 0.05

        img = np.abs(img - noise)

        X[i] = img.flatten()

    return X, y


print("Loading Bars and Stripes Dataset...")

n_features_sqrt = 3
n_qubits = 9

training_size = 100
test_size = 50

X_train, y_train = generate_bars_and_stripes(
    training_size,
    n_features_sqrt
)

X_test, y_test = generate_bars_and_stripes(
    test_size,
    n_features_sqrt
)

X = np.vstack((X_train, X_test))
y = np.concatenate((y_train, y_test))

print("Dataset Loaded")
print("Samples :", len(X))
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
    "bars_stripes_results.csv",
    index=False
)

print("\nResults saved as bars_stripes_results.csv")

# =====================================================
# Average Results
# =====================================================

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
    "bars_stripes_summary.csv",
    index=False
)

print("\nSummary saved as bars_stripes_summary.csv")