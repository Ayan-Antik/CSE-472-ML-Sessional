import numpy as np

print("Enter dimensions of matrix: ")
n = int(input())
A = np.random.randint(0, 100, (n, n))
while np.linalg.det(A) == 0:
    A = np.random.randint(0, 100, (n, n))
print("A = ", A)
eigen_values, eigen_vectors = np.linalg.eig(A)
print("Eigen values = ", eigen_values)
print("Eigen vectors = ", eigen_vectors)
recreate_A = np.dot(eigen_vectors, np.dot(np.diag(eigen_values), np.linalg.inv(eigen_vectors))) 
print("Recreated A = ", recreate_A)
if np.allclose(A, recreate_A):
    print("Reconstruction is ok")
else:
    print("Reconstruction is not ok")
