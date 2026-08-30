import numpy as np
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

# ==========================================================
# Load AdHoc Dataset
# ==========================================================

print("Loading AdHoc Dataset...")

feature_dim = 2
training_size = 100
test_size = 20

X_train, y_train, X_test, y_test = ad_hoc_data(
    training_size=training_size,
    test_size=test_size,
    n=feature_dim,
    gap=0.3,
    plot_data=False,
    one_hot=False
)

# Combine train and test
X = np.vstack((X_train, X_test))
y = np.concatenate((y_train, y_test))

print(f"Total Samples : {len(X)}")

# ==========================================================
# Quantum Kernel
# ==========================================================

sampler = Sampler()

fidelity = ComputeUncompute(
    sampler=sampler
)

feature_map = ZZFeatureMap(
    feature_dimension=feature_dim,
    reps=2,
    entanglement="linear"
)

quantum_kernel = FidelityQuantumKernel(
    fidelity=fidelity,
    feature_map=feature_map
)

# ==========================================================
# Classical Models
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "SVM": SVC(kernel="rbf"),
    "QSVC": QSVC(quantum_kernel=quantum_kernel)
}

# ==========================================================
# 5 Fold Cross Validation
# ==========================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

fold = 1

for train_idx, test_idx in skf.split(X, y):

    print(f"\n========== Fold {fold} ==========")

    X_train_fold = X[train_idx]
    X_test_fold = X[test_idx]

    y_train_fold = y[train_idx]
    y_test_fold = y[test_idx]

    row = {"Fold": fold}

    for name, model in models.items():

        start = time.time()

        model.fit(X_train_fold, y_train_fold)

        prediction = model.predict(X_test_fold)

        elapsed = time.time() - start

        accuracy = accuracy_score(
            y_test_fold,
            prediction
        )

        row[f"{name}_Accuracy"] = accuracy
        row[f"{name}_Time"] = elapsed

        print(
            f"{name:<22}"
            f"Accuracy = {accuracy:.4f}   "
            f"Time = {elapsed:.4f} sec"
        )

    results.append(row)

    fold += 1

# ==========================================================
# Save Results
# ==========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "adhoc_multimodel_results.csv",
    index=False
)

print("\nResults saved as adhoc_multimodel_results.csv")

# ==========================================================
# Average Results
# ==========================================================

print("\n===============================")
print("AVERAGE PERFORMANCE")
print("===============================")

summary = []

for model in models.keys():

    avg_acc = results_df[f"{model}_Accuracy"].mean()
    avg_time = results_df[f"{model}_Time"].mean()

    summary.append(
        {
            "Model": model,
            "Average Accuracy": avg_acc,
            "Average Time (s)": avg_time
        }
    )

summary_df = pd.DataFrame(summary)

print(summary_df)

summary_df.to_csv(
    "adhoc_multimodel_summary.csv",
    index=False
)

print("\nSummary saved as adhoc_multimodel_summary.csv")