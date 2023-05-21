import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA


# sklearn kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples as silhouette

# pyclustering kmeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.encoder import cluster_encoder


# Read CSV files
print('Reading CSV files ...')
M0 = np.genfromtxt("./data/species0_2800-3100cycles.csv", delimiter=',', skip_header=1)
M1 = np.genfromtxt("./data/species0_2600-2900cycles.csv", delimiter=',', skip_header=1)
M3 = np.genfromtxt("./data/species0_2400-2700cycles.csv", delimiter=',', skip_header=1)
M4 = np.genfromtxt("./data/species0_2200-3000cycles.csv", delimiter=',', skip_header=1)
print('Done with reading')

M = np.hstack((M0,M1, M3, M4[0:301,:]))

#Simulation Parameters
Lx = 40.0
Ly = 20.0
n = 40000 #Number of particles to plot 10000

#first electrons exiting after approx 280 cycles, ions are slow, we can use all the cycles
cycles = 300

#extract x and y for plotting: need to check if any particle goes out of box
nx, ny = 0+n*3, 1+n*3
x = M[:cycles,0:nx:3]
y = M[:cycles,1:ny:3]

#disp('Plotting dataset ...');
plt.figure()
plt.plot(x,y,'.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('%s Electron Trajectories During Magnetic Reconnection' %n)
plt.grid()
#plt.show()
#plt.set(gca, 'FontName', 'Times New Roman')
#plt.set(gca, 'FontSize', 16)


# Preprocessing data: apply FFT
# X - FFT
x = x - M[0,0:nx:3]
x = x/np.max(np.absolute(x))
x = np.absolute(np.fft.fft(x, axis=0)/cycles)
#x = abs(fft(x).^2); # not working as good as FFT^2 - compress the differences
x = x/np.max(np.absolute(x))

# Y - FFT
y = y - M[0,1:ny:3]
y = y/np.max(np.absolute(y))
y = np.absolute(np.fft.fft(y, axis=0)/cycles)
#y = abs(fft(y).^2); % not working as good as FFT^2 - compress the differences
y = y/np.max(y)

#energy: we take only mean as energy information is also encoded in the trajectory
energy = M[:cycles,2:(2+n*3):3]
energy = np.mean(energy)



#PCA for dimensionality reduction
num_spectral_modes = int(cycles/2) # can try just with a part of spectrum: cycles/4, cycles/8

traj =np.vstack((x[:num_spectral_modes,:], y[:num_spectral_modes,:]))
pca = PCA()  #(.95) means sk chooses n s.t. 95% of variance is preserved #(n_components=20)
pca.fit(traj.T)
#coefs,score,_,_,explained = pca(traj)

coefs = pca.components_.T
score = pca.transform(traj.T)
explained = pca.explained_variance_ratio_
print(np.shape(coefs))
print(np.shape(explained))
print(np.shape(traj), np.shape(score))

#plot PCA
#plt.fig()
#plt.subplot(2,2,1)
#plt.plot(score[:,1],score[:,2],'.')
#hold on
# %plot(score(1,1),score(1,2),'r*');
# %hold off
# xlabel('PC I')
# ylabel('PC II')
# grid on
# set(gca, 'FontName', 'Times New Roman')
# set(gca, 'FontSize', 18)



# choose the reduced number of PCA components to use in the analysis
num_pca_components = 5
for i in range(20):
    pca_expl = sum(explained[0:i])
    print(pca_expl)
pca_expl = sum(explained[:num_pca_components])

#Clustering
print('Clustering ...')

#for reproducibility
np.random.seed(12)

# k-means -scikit
num_clusters = 25 # need to be square of a number, e.g., 16, 25, 36 or 64
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=50, max_iter=1000, tol=1e-4, algorithm='lloyd')
idx = kmeans.fit_predict(score[:,:num_pca_components], sample_weight=None)
C = kmeans.cluster_centers_

'''
# k-means -pyclustering
def cosine_similarity(point1, point2):
    dimension = len(point1)
    result = 0.0
    for i in range(dimension):
        result += abs(point1[i] - point2[i]) * 0.1
    return result

metric = distance_metric(type_metric.USER_DEFINED, func=cosine_similarity)

initial_centers = random_center_initializer(score, 3, random_state=5).initialize()
instanceKm = kmeans(score, initial_centers=initial_centers, metric=distance_metric(1000))
# perform cluster analysis
instanceKm.process()
# cluster analysis results - clusters and centers
pyClusters = instanceKm.get_clusters()
pyCenters = instanceKm.get_centers()
# enumerate encoding type to index labeling to get labels
pyEncoding = instanceKm.get_cluster_encoding()
pyEncoder = cluster_encoder(pyEncoding, pyClusters, score)
pyLabels = pyEncoder.set_encoding(0).get_clusters()
'''

#plot pca clustered data 
plt.figure()
plt.scatter(score[:,1], score[:,2], num_clusters, idx)
plt.title('K-means -Euclidean Distance')
plt.xlabel('PC1')
plt.ylabel('PC2')

#plot pareto graph
fig, ax = plt.subplots()
ax.bar(range(num_pca_components), explained[:num_pca_components], color="C0")
ax2 = ax.twinx()
ax2.plot([sum(explained[:i])*100 for i in range(num_pca_components)], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")

#count the number of clusters per idx
plt.figure()
counts,_,_ = plt.hist(idx, num_clusters)
plt.xlabel('Cluster ID')
plt.ylabel('#Samples')
plt.grid()
out = np.sort(counts)
ii = np.argsort(counts)

out, ii = out[::-1], ii[::-1]

print('Calculating Silhouette...')
silh2 = silhouette(score[:,:num_pca_components], idx)
bad_samples = np.sum(silh2<0)
#print(silh2)
print(bad_samples)


#plot typical traj for different clusters

plotted_particles=25
#for i in range(num_clusters):
#    xi = x[0:cycles, idx==ii[i] and silh2 > 0.99*np.max(silh2[idx==ii[i]])]






#plt.show()