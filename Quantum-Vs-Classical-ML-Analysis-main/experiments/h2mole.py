import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from qiskit.primitives import Sampler
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVR
from qiskit_machine_learning.state_fidelities import ComputeUncompute


# Distance (in Angstroms)
distances = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 
                      1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 
                      2.4, 2.6, 2.8, 3.0, 3.5, 4.0])
# Energy (Hartree) - The characteristic dissociation curve
energies = np.array([-0.15, -0.60, -0.91, -1.05, -1.11, -1.13, -1.13, -1.12, -1.10, -1.07,
                     -1.05, -1.03, -1.01, -1.00, -0.99, -0.98, -0.97, -0.97, -0.96, -0.96,
                     -0.95, -0.95, -0.94, -0.94, -0.94, -0.94])

X = distances.reshape(-1, 1)
y = energies

# Scale inputs to [-1, 1] for Quantum Gates
scaler_x = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_x.fit_transform(X)

print("Loaded H2 Molecule Dissociation Curve.")
print(f"Data points: {len(X)}")

print("Running Classical SVR (RBF)...")
cml_model = SVR(kernel='rbf', C=10, epsilon=0.01)
cml_model.fit(X_scaled, y)
cml_pred = cml_model.predict(X_scaled)
cml_r2 = r2_score(y, cml_pred)
print(f"CML R2 Score: {cml_r2:.4f}")

print("Running QSVR...")
# ChebyshevFeatureMap - great for fitting continuous functions
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
feature_map = ZFeatureMap(feature_dimension=1, reps=3)

qml_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
qml_model = QSVR(quantum_kernel=qml_kernel, C=10)

qml_model.fit(X_scaled, y)
qml_pred = qml_model.predict(X_scaled)
qml_r2 = r2_score(y, qml_pred)
print(f"QML R2 Score: {qml_r2:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Real Physics Data')
plt.plot(X, cml_pred, color='blue', linestyle='--', label=f'CML SVR (R2={cml_r2:.2f})')
plt.plot(X, qml_pred, color='red', alpha=0.7, label=f'QML QSVR (R2={qml_r2:.2f})')
plt.xlabel("Interatomic Distance (Angstrom)")
plt.ylabel("Energy (Hartree)")
plt.title("H2 Molecule Binding Energy Curve")
plt.legend()
plt.grid(True)
plt.show()