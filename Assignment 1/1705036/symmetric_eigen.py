import numpy as np

print("Enter dimensions of matrix: ")
n = int(input())
A = np.random.randint(0, 100, (n, n))
symmA = A.T*A
# print("A = ", symmA)
while np.linalg.det(symmA) == 0:
    A = np.random.randint(-100, 100, (n, n))
    symmA = A.T*A 
print("A = ", symmA)
eigen_values, eigen_vectors = np.linalg.eig(symmA)
print(f'Eigen values:\n{eigen_values}\n')
# print("Diagonal Eigen values: ", np.diag(eigen_values))
print(f'Eigen vectors:\n{eigen_vectors}\n')
recreate_A = np.dot(eigen_vectors, np.dot(np.diag(eigen_values), eigen_vectors.T))
print(f'Recreated A:\n{recreate_A}\n')
if np.allclose(symmA, recreate_A):
    print("Reconstruction is ok")
else:
    print("Reconstruction is not ok")