import numpy as np

print("Enter dimensions of matrix: ")
print("n = ", end="")
n= int(input())
print("m = ", end="")
m = int(input())
A = np.random.randint(-100, 100, (n, m))
print(f'matrix A: \n{A}\n')
u, d, vh = np.linalg.svd(A, full_matrices=False)
print(u.shape, d.shape, vh.shape)
A_plus = np.linalg.pinv(A)
print(f'A+ \n{A_plus}\n')
print(f'u: \n{u}\nd: \n{d}\nvh: \n{vh}\n')
d = np.diag(np.reciprocal(d))
# print(f'Diagonal d:\n{d}\n')
# if n<m:
#     d = np.concatenate((d, np.zeros((n, m-n))), axis=1)
# else: 
#     d = np.concatenate((d, np.zeros((n-m, m))), axis=0)

recreate_A_plus = np.dot(vh.T, np.dot(d.T, u.T))
print(f'Recreated A+ = \n{recreate_A_plus}')
if np.allclose(A_plus, recreate_A_plus):
    print("Reconstruction is ok")
else:
    print("Reconstruction is not ok")
