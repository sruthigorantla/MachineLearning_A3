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
    with open("syndata.txt","r") as fp:
        for line in fp:
            line = line.split(" ")
            data.append([float(line[0]), float(line[1])])
    data = np.asarray(data)
    with open("syndata_lab.txt","r") as fp:
        for line in fp:
            label.append(int(line[0]))

    plt.scatter(data[:,0],data[:,1])
    plt.show()
    return data, label


def euclidean_distance(x,y):
    return np.sqrt(np.sum((x - y)**2))

def get_k_centroids(points, k):
    centroids = random.sample(range(0, len(points)-1), k)
    return centroids

def kmeans(points, true_labels, n, k):
    iters = 0
    over = False
    centroids = get_k_centroids(points, k)
    clusters = [0]*n
    c = points[centroids]
    dist = [0]*k
    
    while( over is False ):
        iters += 1
        clusters_updated = [0]*n
        for i in range(n):
            dist = np.apply_along_axis(euclidean_distance,0,c,points[i])
            clusters_updated[i] = np.argmin(dist)
        RAND_index.append(adjusted_rand_score(true_labels,clusters_updated))
        over = True
        for i in range(n):
            if clusters_updated[i] != clusters[i]:
                over = False
                break
        if iters > 100:
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

def calculate_kmeans(points, true_labels, k):    
    
    cluster_after_kmeans = kmeans(points, true_labels, len(points), k)
    
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
    # Create plot
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

    points, true_labels = get_data()
    calculate_kmeans(points, true_labels, 2)

if __name__ == "__main__":
    main()