import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
# import scipy.sparse as sp
from sklearn.decomposition import PCA

def Estep(X, p, mix_p):
	N = len(X)
	D = len(X[0])
	K = len(p)
	log_eta = np.zeros((N,K))

	for i in range(N):
		# log_eta[i,:] = (np.log(mix_p)+np.sum(np.log(p[:,X[i,:]]),axis=1).reshape(K,1)+np.sum(np.log(1-p[:,1-X[i,:]]),axis=1).reshape(K,1)).T
		
		for k in range(K):
			sum1 = 0
			sum2 = 0
			for d in range(D):
				sum1 += X[i][d]*np.log(p[k][d])
				sum2 += (1-X[i][d])*np.log(1-p[k][d])
			log_eta[i][k] = np.log(mix_p[k]) + sum1 + sum2
	# log_eta = log_eta/np.sum(log_eta,axis=1).reshape(N,1)
	# eta = np.exp(log_eta - log_eta.max(1).reshape(N,1))
	eta = np.exp(log_eta)
	eta = eta/np.sum(eta,axis=1).reshape(N,1)
	return eta

def Mstep(X, eta, alpha1, alpha2):
	K = len(eta[0])
	N = len(X)
	D = len(X[0])
	p = np.zeros((K,D))

	Nk = np.sum(eta,axis=0).reshape(K,1)
	mix_p = (Nk+alpha2)/(np.sum(Nk)+alpha2*K)
	# print(Nk)
	for k in range(K):
		# p[k,:] = (np.sum(sp.spdiags(eta[:,k],0,N,N)*X)+alpha1)/(alpha1*D+Nk[k])
		sum_k = 0
		for i in range(N):
			sum_k += eta[i][k]*X[i]
		p[k,:] = (sum_k + alpha1)/(alpha1*D+Nk[k])
	# p = np.maximum(p,np.spacing(1))
	# print(p)
	return p, mix_p

def log_likelihood(X, p, mix_p, Z):
	N = len(X)
	D = len(X[0])
	K = len(p)
	log_eta = np.zeros((N,K))

	sum_overall = 0
	for i in range(N):
		# log_eta[i,:] = (np.log(mix_p)+np.sum(np.log(p[:,X[i,:]]),axis=1).reshape(K,1)+np.sum(np.log(1-p[:,1-X[i,:]]),axis=1).reshape(K,1)).T
		
		for k in range(K):
			sum1 = 0
			sum2 = 0
			for d in range(D):
				sum1 += X[i][d]*np.log(p[k][d])
				sum2 += (1-X[i][d])*np.log(1-p[k][d])
			log_eta[i][k] = np.log(mix_p[k]) + sum1 + sum2
		sum_overall += log_eta[i][Z[i]]
	return np.sum(sum_overall)


def MoBlabels(X, p, mix_p):
	alpha1 = alpha2 = 1e-8

	LL = []
	for i in range(1000):
		eta = Estep(X, p, mix_p)
		eta = np.asarray(eta)
		p, mix_p = Mstep(X, eta, alpha1, alpha2)
		Z = np.argmax(eta,axis=1)
		ll = log_likelihood(X, p, mix_p, Z)
		LL.append(ll)
		print("iteration: ",i,ll)
	
	plt.plot(LL)
	plt.show()
	return Z

data = []
with open("MoBern.txt","r") as fp:
	for line in fp:
		data.append([int(x) for x in line.split()])

data = np.asarray(data)
N = data.shape[0]
D = data.shape[1]
K = 2
p = np.random.uniform(0,1,(K,D))
p /= np.sum(p,axis=1).reshape(K,1)
mix_p = np.zeros((K,1))
for i in range(K):
	mix_p[i][0] = 1/(i+1)
mix_p /= np.sum(mix_p,axis=0)
clusters = MoBlabels(data, p, mix_p)


pca = PCA(n_components=2)
pca.fit(data)
points_2d = pca.transform(data)
g = []

for i in range(K):
    g.append(([],[]))
# print(g)

for i in range(N):
    for j in range(K):
        if(clusters[i] == j):
            g[j][0].append(points_2d[i][0])
            g[j][1].append(points_2d[i][1])

data = []
for i in range(K):
    data.append((np.asarray(g[i][0]),np.asarray(g[i][1])))
data = tuple(data)

colors = ("red", "green")
# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for data, color in zip(data, colors):
    x, y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)
 
plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()
