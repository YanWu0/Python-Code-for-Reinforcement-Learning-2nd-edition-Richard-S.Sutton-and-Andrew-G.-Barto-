import numpy as np
import random

row_num = 5
col_num = 5
A_pos = np.array([0,1])
A_prime_pos = np.array([4,1])
B_pos = np.array([0,3])
B_prime_pos = np.array([2,3])
gamma = 0.9

def transition_mat_and_reward(row_num, col_num, A_pos, A_prime_pos, B_pos, B_prime_pos):
    n = row_num*col_num
    tra_mat = np.zeros((n,n))
    reward = np.zeros(n)
    for i in range(row_num):
        for j in range(col_num):
            w = 0
            if 0 <= i-1:
                tra_mat[i*col_num+j, (i-1)*col_num+j] = 1/4
            else:
                w += 1
            if i+1 < row_num:
                tra_mat[i*col_num+j, (i+1)*col_num+j] = 1/4
            else:
                w += 1
            if 0 <= j-1:
                tra_mat[i*col_num+j, i*col_num+j-1] = 1/4
            else:
                w += 1
            if j+1 < col_num:
                tra_mat[i*col_num+j, i*col_num+j+1] = 1/4
            else:
                w += 1
            tra_mat[i*col_num+j,i*col_num+j] = 1/4 * w
            reward[i*col_num+j] = -0.25*w
    a_r,a_c = A_pos
    ap_r, ap_c = A_prime_pos
    b_r,b_c = B_pos
    bp_r, bp_c = B_prime_pos
    tra_mat[a_r * col_num + a_c,:] = 0
    tra_mat[b_r * col_num + b_c,:] = 0
    tra_mat[a_r*col_num+a_c, ap_r*col_num+ap_c] = 1
    tra_mat[b_r*col_num+b_c, bp_r*col_num+bp_c] = 1
    reward[a_r*col_num+a_c] = 10
    reward[b_r*col_num+b_c] = 5
    return tra_mat, reward

# v is the state value of uniformly random policy, in example 3.5
p, r = transition_mat_and_reward(row_num, col_num, A_pos, A_prime_pos, B_pos, B_prime_pos)
I = np.identity(row_num*col_num)
v = np.linalg.inv(I - gamma * p).dot(r)
print(v)
