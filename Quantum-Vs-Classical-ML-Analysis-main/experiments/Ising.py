import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

# -------------------------------------------------------------------
#  STEP 1: SYNTHESIZE ISING MODEL DATA
# -------------------------------------------------------------------
def generate_ising_data(n_samples, n_spins):
    """
    Class 0: Ordered (Ferromagnetic) - Spins strongly correlated.
    Class 1: Disordered (Paramagnetic) - Spins random.
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        label = np.random.randint(2)
        y.append(label)
        
        if label == 0: # Ordered Phase (Low Temp)
            # Pick a direction (All Up or All Down)
            base_state = np.random.randint(2) 
            # Create state with very low probability of flipping
            spins = [base_state if np.random.rand() > 0.1 else 1-base_state for _ in range(n_spins)]
        else: # Disordered Phase (High Temp)
            # Totally random spins
            spins = np.random.randint(2, size=n_spins)
            
        X.append(spins)
        
    return np.array(X), np.array(y)

print("Generating Ising Model data (9 Spins)...")
n_qubits = 9
n_samples = 200

X, y = generate_ising_data(n_samples, n_qubits)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Data prepared: {len(X_train)} samples.")
target_names_ising = ["Ordered (Ferro)", "Disordered (Para)"]

print("Running Classical SVM...")
cml_model = SVC(kernel='rbf')
cml_model.fit(X_train, y_train)
cml_acc = accuracy_score(y_test, cml_model.predict(X_test))

print(f"CML Accuracy: {cml_acc:.4f}")

print(f"Running QSVC (ZZFeatureMap, {n_qubits} qubits)...")


sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)

# The ZZ interaction physically matches the Ising model Hamiltonian
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')

qml_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
qml_model = QSVC(quantum_kernel=qml_kernel)

start = time.time()
qml_model.fit(X_train, y_train)
train_time = time.time() - start

qml_acc = accuracy_score(y_test, qml_model.predict(X_test))

print(f"QML Accuracy: {qml_acc:.4f}")
print(f"Training Time: {train_time:.2f}s")
