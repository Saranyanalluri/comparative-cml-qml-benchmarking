import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

print("Loading and preparing Iris dataset...")

iris = load_iris()
X = iris.data
y = iris.target

# Use 2 classes (Setosa=0 and Versicolor=1), remove Virginica=2
mask = y != 2
X = X[mask]
y = y[mask]

# Use 2 features (sepal length, sepal width)
n_qubits = 2
X = X[:, :n_qubits]


training_sizes = [10, 20, 30, 40, 50, 60]

cml_scores = []
qml_scores = []



for size in training_sizes:
    print(f"\n--- Training Size: {size} ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=size, random_state=42, stratify=y
    )

    # Scale the data (Crucial for QSVC)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- CML (SVC) ---
    cml_model = SVC(kernel='rbf')
    cml_model.fit(X_train_scaled, y_train)
    cml_pred = cml_model.predict(X_test_scaled)
    cml_acc = accuracy_score(y_test, cml_pred)
    cml_scores.append(cml_acc)
    print(f"CML Accuracy: {cml_acc:.2f}")

    # --- QML (QSVC) ---
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
    qml_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    
    qml_model = QSVC(quantum_kernel=qml_kernel)
    qml_model.fit(X_train_scaled, y_train)
    qml_pred = qml_model.predict(X_test_scaled)
    qml_acc = accuracy_score(y_test, qml_pred)
    qml_scores.append(qml_acc)
    print(f"QML Accuracy: {qml_acc:.2f}")


plt.figure(figsize=(10, 6))
plt.plot(training_sizes, cml_scores, marker='o', linestyle='-', color='blue', label='CML (SVC-RBF)')
plt.plot(training_sizes, qml_scores, marker='s', linestyle='--', color='green', label='QML (QSVC)')

plt.title('CML vs QML Accuracy on Iris Dataset (Setosa vs Versicolor)')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.ylim(0.5, 1.05) 
plt.grid(True)
plt.legend()

plt.savefig('iris_comparison_plot.png')
plt.show()