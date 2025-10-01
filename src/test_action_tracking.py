import numpy as np
from scipy.linalg import solve_triangular


n = 6

x_traget = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).T
w = np.array([0.01, 0.01, 0.4, 0.01, 0.01, 0.01])
w = w / np.sum(w)
past_actions = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) * 0.5

A = np.zeros((n, n))
for kk in range(n):
    try:
        w_line = np.fliplr(np.expand_dims(w[:kk+1],0))
    except:
        w_line = w[:kk+1]
    A[kk, :kk+1] = w_line


B = np.zeros((n, n))
for kk in range(n):
    B[kk, 1+kk:] = past_actions[kk:]

b = B @ w.T

solution = solve_triangular(A, x_traget - b, lower=True)


print('w = ', w)
print('A = ', A)
print('B = ', B)
print('b = ', b)
print('solution =', [f'{x:.2f}' for x in solution])



