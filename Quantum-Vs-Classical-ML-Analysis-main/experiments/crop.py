import numpy as np
import pandas as pd
import time
import os

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

print("=" * 60)
print("Crop Recommendation Dataset Benchmark")
print("=" * 60)

# =====================================================
# Load Dataset
# =====================================================

dataset_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "Crop_recommendation.csv"
)

df = pd.read_csv(dataset_path)

print("\nDataset Loaded Successfully")
print(df.head())

# =====================================================
# Dataset Information
# =====================================================

print("\nDataset Shape :", df.shape)
print("Missing Values :", df.isnull().sum().sum())

# =====================================================
# Features & Labels
# =====================================================

X = df.drop("label", axis=1).values
y = df["label"].values

print(f"\nOriginal Samples : {len(X)}")
print(f"Number of Features : {X.shape[1]}")

# =====================================================
# Encode Labels
# =====================================================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print(f"Number of Classes : {len(np.unique(y))}")

# =====================================================
# Reduce Dataset Size
# (QSVC becomes very slow on all 2200 samples)
# =====================================================

X, _, y, _ = train_test_split(
    X,
    y,
    train_size=500,
    stratify=y,
    random_state=42
)

print(f"\nSamples Used For Benchmark : {len(X)}")

# =====================================================
# Feature Scaling
# =====================================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("\nFeature Scaling Completed.")

# =====================================================
# Cross Validation
# =====================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =====================================================
# Store Accuracy
# =====================================================

lr_scores = []
knn_scores = []
rf_scores = []
svm_scores = []
qsvc_scores = []
pegasos_scores = []

# =====================================================
# Store Runtime
# =====================================================

lr_times = []
knn_times = []
rf_times = []
svm_times = []
qsvc_times = []
pegasos_times = []

print("\nInitialization Completed.")
print("=" * 60)
print("Starting 5-Fold Cross Validation...")
print("=" * 60)

# =====================================================
# Cross Validation Loop Starts Here
# =====================================================

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

    print(f"\n========== Fold {fold+1} ==========")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]
        # =====================================================
    # Logistic Regression
    # =====================================================

    lr_model = LogisticRegression(max_iter=1000)

    start = time.time()

    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    lr_time = time.time() - start

    lr_acc = accuracy_score(y_test, lr_pred)

    lr_scores.append(lr_acc)
    lr_times.append(lr_time)

    print(f"LR Accuracy : {lr_acc:.4f}")

    # =====================================================
    # KNN
    # =====================================================

    knn_model = KNeighborsClassifier(
        n_neighbors=5
    )

    start = time.time()

    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)

    knn_time = time.time() - start

    knn_acc = accuracy_score(y_test, knn_pred)

    knn_scores.append(knn_acc)
    knn_times.append(knn_time)

    print(f"KNN Accuracy : {knn_acc:.4f}")

    # =====================================================
    # Random Forest
    # =====================================================

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    start = time.time()

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_time = time.time() - start

    rf_acc = accuracy_score(y_test, rf_pred)

    rf_scores.append(rf_acc)
    rf_times.append(rf_time)

    print(f"RF Accuracy : {rf_acc:.4f}")

    # =====================================================
    # SVM
    # =====================================================

    svm_model = SVC(
        kernel="rbf"
    )

    start = time.time()

    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)

    svm_time = time.time() - start

    svm_acc = accuracy_score(y_test, svm_pred)

    svm_scores.append(svm_acc)
    svm_times.append(svm_time)

    print(f"SVM Accuracy : {svm_acc:.4f}")

    # =====================================================
    # Quantum Kernel
    # =====================================================

    sampler = StatevectorSampler()

    fidelity = ComputeUncompute(
        sampler=sampler
    )

    feature_map = ZZFeatureMap(
        feature_dimension=X.shape[1],
        reps=2,
        entanglement="linear"
    )

    quantum_kernel = FidelityQuantumKernel(
        fidelity=fidelity,
        feature_map=feature_map
    )

    # =====================================================
    # QSVC
    # =====================================================

    qsvc_model = QSVC(
        quantum_kernel=quantum_kernel
    )

    start = time.time()

    qsvc_model.fit(X_train, y_train)
    qsvc_pred = qsvc_model.predict(X_test)

    qsvc_time = time.time() - start

    qsvc_acc = accuracy_score(y_test, qsvc_pred)

    qsvc_scores.append(qsvc_acc)
    qsvc_times.append(qsvc_time)

    print(f"QSVC Accuracy : {qsvc_acc:.4f}")
    # =====================================================
# Final Results
# =====================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"Logistic Regression : {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")
print(f"KNN                 : {np.mean(knn_scores):.4f} ± {np.std(knn_scores):.4f}")
print(f"Random Forest       : {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
print(f"SVM                 : {np.mean(svm_scores):.4f} ± {np.std(svm_scores):.4f}")
print(f"QSVC                : {np.mean(qsvc_scores):.4f} ± {np.std(qsvc_scores):.4f}")


print("\n" + "=" * 70)
print("AVERAGE RUNTIME")
print("=" * 70)

print(f"Logistic Regression : {np.mean(lr_times):.4f} sec")
print(f"KNN                 : {np.mean(knn_times):.4f} sec")
print(f"Random Forest       : {np.mean(rf_times):.4f} sec")
print(f"SVM                 : {np.mean(svm_times):.4f} sec")
print(f"QSVC                : {np.mean(qsvc_times):.4f} sec")


# =====================================================
# Paired t-tests
# =====================================================

print("\n" + "=" * 70)
print("PAIRED T-TEST")
print("=" * 70)

t_stat, p_value = ttest_rel(svm_scores, qsvc_scores)

print("\nSVM vs QSVC")
print(f"t-statistic : {t_stat:.4f}")
print(f"p-value     : {p_value:.6f}")

t_stat, p_value = ttest_rel(svm_scores, pegasos_scores)

print("\nSVM vs PegasosQSVC")
print(f"t-statistic : {t_stat:.4f}")
print(f"p-value     : {p_value:.6f}")

# =====================================================
# Save Results
# =====================================================

results_df = pd.DataFrame({

    "LR_Accuracy": lr_scores,
    "KNN_Accuracy": knn_scores,
    "RF_Accuracy": rf_scores,
    "SVM_Accuracy": svm_scores,
    "QSVC_Accuracy": qsvc_scores,


    "LR_Time": lr_times,
    "KNN_Time": knn_times,
    "RF_Time": rf_times,
    "SVM_Time": svm_times,
    "QSVC_Time": qsvc_times,
   

})

results_df.to_csv(
    "crop_multimodel_results.csv",
    index=False
)

print("\nResults saved to crop_multimodel_results.csv")

# =====================================================
# Best Model
# =====================================================

avg_scores = {
    "Logistic Regression": np.mean(lr_scores),
    "KNN": np.mean(knn_scores),
    "Random Forest": np.mean(rf_scores),
    "SVM": np.mean(svm_scores),
    "QSVC": np.mean(qsvc_scores),
   
}

best_model = max(avg_scores, key=avg_scores.get)

print("\n" + "=" * 70)
print(f"Best Performing Model : {best_model}")
print(f"Average Accuracy      : {avg_scores[best_model]:.4f}")
print("=" * 70)