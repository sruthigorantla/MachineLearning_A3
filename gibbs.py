import numpy as np

np.random.seed(15190)

W = np.random.normal(0,1,(10,20))
np.random.seed(15190)
b = np.random.normal(0,1,(10,1))
np.random.seed(15190)
c = np.random.normal(0,1,(20,1))
N = 50

np.random.seed(1049*16)
V = np.random.randint(0,2,size = 10).reshape(-1,1)
np.random.seed(1049*16)
H = np.random.randint(0,2,size = 20).reshape(-1,1)

hid = np.zeros((N, 20))
vis = np.zeros((N, 10))

def sigmoid(u):
    return 1.0/(1 + np.exp(-u))

def gibbs_sampling(W, c, b, V):
    l = np.matmul(V.T, W) + c.T
    f_s = sigmoid(l + c.T)
    h_new = np.random.binomial(size=f_s.shape, n=1, p=f_s)
    f_u = sigmoid(np.dot(h_new, W.transpose()) + b.T)
    v_new = np.random.binomial(size=f_u.shape, n=1, p=f_u)
    return [f_s, h_new, f_u, v_new]

f_s, hid[0], f_u, vis[0] = gibbs_sampling(W, c, b, V) 

for t in range(1,N):
    f_s, hid[t], f_u, vis[t] = gibbs_sampling(W, c, b, vis[t-1])
    
    
print(vis[-1]) 