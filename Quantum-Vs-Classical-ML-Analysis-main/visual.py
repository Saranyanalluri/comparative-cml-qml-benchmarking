import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from qiskit_machine_learning.datasets import ad_hoc_data

iris = load_iris()
X = iris.data
y = iris.target

# Filter 
X_simple = X[y != 2][:, :2]
y_simple = y[y != 2]

plt.figure(figsize=(6, 6))
plt.scatter(X_simple[y_simple == 0][:, 0], X_simple[y_simple == 0][:, 1],
            label="Setosa (Class 0)", marker='o')
plt.scatter(X_simple[y_simple == 1][:, 0], X_simple[y_simple == 1][:, 1],
            label="Versicolor (Class 1)", marker='x')

plt.title("Simplified Iris Dataset (2 Classes, 2 Features)")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()
plt.grid(True)
plt.show()
feature_dim = 2
training_size = 20
test_size = 10

X_train, y_train, X_test, y_test = ad_hoc_data(
    training_size=training_size,
    test_size=test_size,
    n=feature_dim,
    gap=0.3, # This gap influences separation
    plot_data=False # We'll plot it ourselves
)


print("Shape of y_train:", X_train.shape)
print("Shape of y_test:", X_test.shape)


X_adhoc = np.vstack((X_train, X_test))
y_adhoc = np.vstack((y_train, y_test))


plt.figure(figsize=(6, 6))


mask_A = y_adhoc[:, 0] == 1 
plt.scatter(X_adhoc[mask_A, 0],  # Get feature 1 for Class A rows
            X_adhoc[mask_A, 1],  # Get feature 2 for Class A rows
            label="Class A (0)", marker='o')

mask_B = y_adhoc[:, 1] == 1 
plt.scatter(X_adhoc[mask_B, 0],  # Get feature 1 for Class B rows
            X_adhoc[mask_B, 1],  # Get feature 2 for Class B rows
            label="Class B (1)", marker='x')

plt.title("Qiskit Ad Hoc Dataset (2 Features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()