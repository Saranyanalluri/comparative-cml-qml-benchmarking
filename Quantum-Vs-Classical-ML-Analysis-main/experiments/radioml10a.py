import os
import pickle
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel

from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

print("=" * 60)
print("RadioML 2016.10A Multi-Model Benchmark")
print("=" * 60)

# ============================================================
# Load Dataset
# ============================================================

dataset_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "RML2016.10a_dict.dat"
)

with open(dataset_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("Dataset Loaded Successfully")

# ============================================================
# Feature Extraction
# ============================================================

def extract_features(signal):

    I = signal[0]
    Q = signal[1]

    features = [

        np.mean(I),
        np.mean(Q),

        np.std(I),
        np.std(Q),

        np.max(I),
        np.max(Q),

        np.min(I),
        np.min(Q),

        np.sqrt(np.mean(I ** 2)),
        np.sqrt(np.mean(Q ** 2)),

        np.mean(I ** 2),
        np.mean(Q ** 2),

        np.var(I),
        np.var(Q)

    ]

    return features

# ============================================================
# Dataset Selection
# ============================================================

MODULATIONS = [
    "BPSK",
    "QPSK",
    "8PSK",
    "QAM16"
]

SNR = 18

X = []
y = []

for modulation in MODULATIONS:

    key = (modulation, SNR)

    if key not in data:
        continue

    signals = data[key]

    print(f"{modulation}: {len(signals)} samples")

    for signal in signals:

        X.append(extract_features(signal))
        y.append(modulation)

X = np.array(X)
y = np.array(y)

print("\nTotal Samples :", len(X))
print("Original Features :", X.shape[1])

# ============================================================
# Preprocessing
# ============================================================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ============================================================
# PCA for Quantum Models Only
# ============================================================

pca = PCA(n_components=6)

X_qml = pca.fit_transform(X)

print("Classical Features :", X.shape[1])
print("Quantum Features   :", X_qml.shape[1])

# ============================================================
# Cross Validation
# ============================================================

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

print("\nInitialization Completed.")
print("\nStarting 5-Fold Cross Validation...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

    print(f"\n========== Fold {fold+1} ==========")

    # --------------------------------------------------
    # Classical Data
    # --------------------------------------------------

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # --------------------------------------------------
    # Quantum Data (PCA)
    # --------------------------------------------------

    X_train_qml = X_qml[train_idx]
    X_test_qml = X_qml[test_idx]

    # Reduce training samples only for QSVC
    X_train_qml, _, y_train_qml, _ = train_test_split(
        X_train_qml,
        y_train,
        train_size=150,
        stratify=y_train,
        random_state=42
    )

    # ==================================================
    # Logistic Regression
    # ==================================================

    model = LogisticRegression(max_iter=1000)

    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    elapsed = time.time() - start

    lr_scores.append(accuracy_score(y_test, pred))
    lr_times.append(elapsed)

    print(f"LR Accuracy   : {lr_scores[-1]:.4f}")

    # ==================================================
    # KNN
    # ==================================================

    model = KNeighborsClassifier(n_neighbors=5)

    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    elapsed = time.time() - start

    knn_scores.append(accuracy_score(y_test, pred))
    knn_times.append(elapsed)

    print(f"KNN Accuracy  : {knn_scores[-1]:.4f}")

    # ==================================================
    # Random Forest
    # ==================================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    elapsed = time.time() - start

    rf_scores.append(accuracy_score(y_test, pred))
    rf_times.append(elapsed)

    print(f"RF Accuracy   : {rf_scores[-1]:.4f}")

    # ==================================================
    # Classical SVM
    # ==================================================

    model = SVC(kernel="rbf")

    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    elapsed = time.time() - start

    svm_scores.append(accuracy_score(y_test, pred))
    svm_times.append(elapsed)

    print(f"SVM Accuracy  : {svm_scores[-1]:.4f}")

    # ==================================================
    # Quantum Kernel
    # ==================================================

    sampler = StatevectorSampler()

    fidelity = ComputeUncompute(
        sampler=sampler
    )

    feature_map = ZZFeatureMap(
        feature_dimension=6,
        reps=2,
        entanglement="linear"
    )

    quantum_kernel = FidelityQuantumKernel(
        fidelity=fidelity,
        feature_map=feature_map
    )

    # ==================================================
    # QSVC
    # ==================================================

    model = QSVC(
        quantum_kernel=quantum_kernel
    )

    start = time.time()

    model.fit(
        X_train_qml,
        y_train_qml
    )

    pred = model.predict(
        X_test_qml
    )

    elapsed = time.time() - start

    qsvc_scores.append(
        accuracy_score(
            y_test,
            pred
        )
    )

    qsvc_times.append(elapsed)

    print(f"QSVC Accuracy : {qsvc_scores[-1]:.4f}")

    # ==================================================
    # PegasosQSVC
    # ==================================================

    pegasos_scores.append(np.nan)
    pegasos_times.append(np.nan)

    print("PegasosQSVC   : Skipped (Multiclass Dataset)")
    # ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Logistic Regression : {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")
print(f"KNN                 : {np.mean(knn_scores):.4f} ± {np.std(knn_scores):.4f}")
print(f"Random Forest       : {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
print(f"SVM                 : {np.mean(svm_scores):.4f} ± {np.std(svm_scores):.4f}")
print(f"QSVC                : {np.mean(qsvc_scores):.4f} ± {np.std(qsvc_scores):.4f}")

print("\n" + "=" * 60)
print("AVERAGE RUNTIME")
print("=" * 60)

print(f"Logistic Regression : {np.mean(lr_times):.4f} sec")
print(f"KNN                 : {np.mean(knn_times):.4f} sec")
print(f"Random Forest       : {np.mean(rf_times):.4f} sec")
print(f"SVM                 : {np.mean(svm_times):.4f} sec")
print(f"QSVC                : {np.mean(qsvc_times):.4f} sec")

# ============================================================
# PAIRED T-TEST
# ============================================================

print("\n" + "=" * 60)
print("PAIRED T-TEST")
print("=" * 60)

t_stat, p_value = ttest_rel(svm_scores, qsvc_scores)

print("\nSVM vs QSVC")
print(f"t-statistic : {t_stat:.4f}")
print(f"p-value     : {p_value:.6f}")

results_df = pd.DataFrame({

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

results_df.to_csv(
    "radioml_multimodel_results.csv",
    index=False
)

print("\nResults saved to radioml_multimodel_results.csv")
summary = pd.DataFrame({

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

    "Average Runtime (s)": [
        np.mean(lr_times),
        np.mean(knn_times),
        np.mean(rf_times),
        np.mean(svm_times),
        np.mean(qsvc_times)
    ]

})

summary.to_csv(
    "radioml_summary.csv",
    index=False
)

print("Summary saved to radioml_summary.csv")