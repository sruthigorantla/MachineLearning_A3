import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import os.path
import random
import math
import sys
import csv

RAND_index = []

def get_data():
    data = []
    label = []
    M = []
    with open("syndata.txt","r") as fp:
        for line in fp:
            line = line.split(" ")
            data.append([float(line[0]), float(line[1])])
    data = np.asarray(data)
    with open("syndata_kernel.txt","r") as fp:
        for line in fp:
            line = line.split(" ")
            M.append([float(str(line[j])) for j in range(len(line)-1)])
    M = np.asarray(M)
    with open("syndata_lab.txt","r") as fp:
        for line in fp:
            label.append(int(line[0]))

    plt.scatter(data[:,0],data[:,1])
    plt.show()
    return data, label, M


def distance(i,centroids,M):
    return 1.0/M[i,centroids]

def get_k_centroids(points, k):
    centroids = random.sample(range(0, len(points)-1), k)
    return centroids

def kmeans(points, true_labels, M, n, k):
    iters = 0
    over = False
    centroids = get_k_centroids(points, k)
    clusters = [0]*n
    c = [0]*k
    dist = [0]*k
    for i in range(k):
        c[i] = points[centroids[i]]
    
    while( over is False ):
        iters += 1
        clusters_updated = [0]*n
        dist = np.zeros((n,k))
        cluster_points = []
        for i in range(k):
            cluster_k = []
            for j in range(n):
                if(clusters[j] == i):
                    cluster_k.append(j)
            cluster_points.append(cluster_k)
        for i in range(n):
            for j in range(k):
                if(iters > 1):
                    avg = 0
                    for l in range(len(cluster_points[j])):
                        avg += 1.0/M[i][cluster_points[j][l]]
                    avg /= len(cluster_points[j])
                    dist[i][j] = avg
                else:
                    dist[i][j] = 1.0/M[i][centroids[j]]
        clusters_updated = np.argmin(dist,axis=1)
        RAND_index.append(adjusted_rand_score(true_labels,clusters_updated))
        over = True
        for i in range(n):
            if clusters_updated[i] != clusters[i]:
                over = False
                break
        if iters > 200:
            over = True
        if not over:
            x = [0]*k
            y = [0]*k
            count = [0]*k

            for i in range(n):
                for j in range(k):
                    if( clusters_updated[i] == j):
                        count[j] += 1
                        x[j] += points[i][0]
                        y[j] += points[i][1]
            for i in range(k):
                x[i] /= count[i]
                y[i] /= count[i]
                c[i] = [x[i], y[i]]
            clusters = clusters_updated

            
    return clusters   

def min(a,b):
    if a < b:
        return a
    else:
        return b

def calculate_kmeans(points, true_labels, M, k):    
    
    cluster_after_kmeans = kmeans(points, true_labels, M, len(points), k)
    
    N = len(points)
    g = []

    for i in range(k):
        g.append(([],[]))
    # print(g)
    for i in range(N):
        for j in range(k):
            if(cluster_after_kmeans[i] == j):
                g[j][0].append(points[i][0])
                g[j][1].append(points[i][1])

    data = []
    for i in range(k):
        data.append((np.asarray(g[i][0]),np.asarray(g[i][1])))
    data = tuple(data)
    colors = ("red", "green", "blue", "yellow", "pink")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
 
    for data, color in zip(data, colors):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)
     
    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()

    plt.plot(RAND_index)
    plt.show()
    
def main():

    points, true_labels, M = get_data()
    calculate_kmeans(points, true_labels, M, 5)

if __name__ == "__main__":
    main()