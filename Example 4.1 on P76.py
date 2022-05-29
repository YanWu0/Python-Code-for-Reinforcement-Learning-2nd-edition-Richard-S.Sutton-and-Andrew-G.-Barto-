
#######################################################################################################
#  D  1  2  3
#  4  5  6  7
#  8  9  10 11
#  12 13 14 D
# random policy, D = door, undiscounted, episodic,
# r=-1 for each step, V(D)=0
# use iterative policy evaluation to update state values
# v_k+1(s) = Sum_a pi(a|s) Sum_{s',r}p(s',r|s,a)[r + gamma* v_k(s')]
# pi(a|s)=1/4, p(s',r|s,a)=1 or 0, r=-1, gamma = 1
# formula simplification:
# v(1) = 1/4 * ( -1 +v(1)) + 1/4 * (-1 + v(2)) + 1/4 * (-1 + v(5)) + 1/4 * (-1 + v(0))
#      = 1/4 * (-4 + v(1) + v(2) + v(5) + v(0))
#      = -1 + 1/4 (v(1) + v(2) + v(5) + v(0))
# iterative policy evaluation有两种方法
#（1） 计算出全部state的新的value之后一起更新
#（2） 计算出一个state的新的value就更新一个，在计算下一个的时候就可以用这个state valuele
# 这两种方法都收敛到true state value，但是第二种convergence rate 更大， 这个code用的是第二种，有交'in place'
# 这个算法是成功的，与书上P77的value是一致的，但是k=1，2，10 的时候有区别，k=infinity的时候是约等的。
#######################################################################################################
import numpy as np
import matplotlib.pyplot as plt

row_num = 4
col_num = 4
theta = 1e-4

# action : 0=up, 1=right, 2=down, 3=left
# now use this function to create a list of list of list:
# shows next 4 state for a given current_state
def next_state_collection(row_num, col_num):
    next_state = []
    for r in range(row_num):
        for c in range(col_num):
            next_state.append([])
            for i in range(4):
                if i == 0:
                    tem_state = [r-1, c]
                elif i == 2:
                    tem_state = [r, c+1]
                elif i == 3:
                    tem_state = [r+1, c]
                else:
                    tem_state = [r, c-1]
                tem_r, tem_c = tem_state
                if tem_r < 0 or tem_r >= row_num or tem_c <0 or tem_c >= col_num:
                    next_state[-1].append([r,c])
                else:
                    next_state[-1].append(tem_state)
    return next_state
# print(next_state_collection(4,4))

def updata(init_V, row_num, col_num, theta):
    V = init_V
    next_state_coll = next_state_collection(row_num, col_num)
    delta_list = []
    delta = theta
    n = 0
    while delta >= theta:
        n += 1
        delta = 0
        for r in range(row_num):
            for c in range(col_num):
                if [r,c] != [0,0] and [r,c] != [row_num-1, col_num-1]:
                    D1_pos = r * col_num + c
                    sum_next_v = 0
                    for i in range(4):
                        x,y = next_state_coll[D1_pos][i]
                        sum_next_v += V[x,y]
                    new_v = -1 + 1/4 * sum_next_v
                    delta = max(delta, abs(new_v-V[r,c]))
                    V[r,c] = new_v
        delta_list.append(delta)
        if n == 1 or n == 2 or n == 3 or n == 10 or n == 9 or n == 11:
            print(str(n)+' iter value is:')
            print(V)
    return delta_list, V

V = np.zeros((4,4))
delta_list , V = updata(V,row_num, col_num, theta)

n = len(delta_list)
print(n)
print(V)

plt.plot(range(n),delta_list,'m--')
plt.xlabel("iter_num")
plt.ylabel("delta")
plt.show()
