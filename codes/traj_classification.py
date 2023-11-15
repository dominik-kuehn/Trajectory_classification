import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA

# sklearn kmeans
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples as silhouette


title_font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }



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

mode = int(cycles/2)

#extract x and y for plotting: need to check if any particle goes out of box
nx, ny, ne = 1+n*3, 2+n*3, 3+n*3
x_original = M[:cycles,0:nx:3]
y_original = M[:cycles,1:ny:3]
energy_original = M[:cycles,2:ne:3]

'''
#disp('Plotting dataset ...');
plt.figure()
plt.plot(x,y,'.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('%s Electron Trajectories During Magnetic Reconnection' %n, fontdict=title_font)
plt.grid()
'''


# Preprocessing data: apply FFT
# X - FFT
x = x_original - M[0,0:nx:3]
x = x/np.max(np.absolute(x), axis=0)
x = np.absolute(np.fft.fft(x, axis=0))
x = x/np.max(x, axis=0)

# Y - FFT
y = y_original - M[0,1:ny:3]
y = y/np.max(np.absolute(y), axis=0)
y = np.absolute(np.fft.fft(y, axis=0))
y = y/np.max(y, axis=0)

#energy: we take only mean as energy information is also encoded in the trajectory
energy = energy_original #- M[0,2:ne:3]
energy = energy/np.max(np.absolute(energy), axis=0)
#energy = np.absolute(np.fft.fft(energy, axis=0))
#energy = energy/np.max(energy, axis=0)


#PCA for dimensionality reduction
num_spectral_modes = mode#int(cycles/2) # can try just with a part of spectrum: cycles/4, cycles/8

traj =np.vstack((x[:num_spectral_modes,:], y[:num_spectral_modes,:], energy))
pca = PCA()  #(.95) means sk chooses n s.t. 95% of variance is preserved #(n_components=20)
pca.fit(traj.T)

coefs = pca.components_.T
score = pca.transform(traj.T)
explained = pca.explained_variance_ratio_


# choose the reduced number of PCA components to use in the analysis
num_pca_components = 20
#if num_pca_components > mode*2:
#     num_pca_components = mode*2

pca_expl_arr = []
for i in range(num_pca_components):
    pca_expl = sum(explained[0:i+1])
    pca_expl_arr.append(pca_expl)
    print(pca_expl)
pca_expl = sum(explained[:num_pca_components])

#Clustering
print('Clustering ...')

#for reproducibility
np.random.seed(12)


#plot_file_names
traj_clusters_plot_name = "traj_clusters_plot_w_energy"
pca_cluster_plot_name = "pca_clusters_plot_w_energy"
pareto_plot_name = "pareto_plot_pca_w_energy"


# k-means -scikit
num_clusters = 12 # need to be square of a number, e.g., 16, 25, 36 or 64
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=50, max_iter=1000, tol=1e-4, algorithm='lloyd')


### this part doesnt actually do anything
def euc_dist(X, Y = None, Y_norm_squared = None, squared = False):
        #return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
        return cosine_similarity(X, Y)
kmeans.euclidean_distances = euc_dist
###


idx = kmeans.fit_predict(score[:,:num_pca_components], sample_weight=None)
C = kmeans.cluster_centers_


#plot pca clustered data 
plt.figure()
plt.scatter(score[:,0], score[:,1], num_clusters, idx)
plt.title('K-means -Euclidean Distance', fontdict=title_font)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(pca_cluster_plot_name, dpi=150)

#plot pareto graph
fig, ax = plt.subplots()
ax.bar(range(num_pca_components), 100*explained[:num_pca_components], color="C0")
ax2 = ax.twinx()
ax2.plot([sum(explained[:i])*100 for i in range(num_pca_components)], color="C1", marker="D", ms=7)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.xaxis.get_major_locator().set_params(integer=True)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
ax.set_xlabel('Principal Components')
ax.set_ylabel('% of total variance per PC', color="C0")
ax2.set_ylabel('Accumulated % of total variance', color="C1")
fig.savefig(pareto_plot_name, dpi=150)

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

print(bad_samples)


#plot typical traj for different clusters
x = M[:cycles,0:nx:3]
y = M[:cycles,1:ny:3]

plotted_particles=25
fig = plt.figure(figsize=(10,11))
for i in range(num_clusters):
    silh_i, x_i, y_i  = silh2[idx==ii[i]], x[:, idx==ii[i]], y[:, idx==ii[i]]   #selecting sil-score, x and y of right class
    silh_topindex = np.array(np.argsort(silh_i))[::-1]
    xi = x_i[:, silh_topindex[:plotted_particles]]
    yi = y_i[:, silh_topindex[:plotted_particles]]
    plt.subplot(4,3,i+1)
    plt.plot(xi, yi)
    perc = counts[ii]/n*100
    plt.title("%d: %3.1f"%(i, perc[i]), fontdict=title_font)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.axis([12,28,8,12])
    #plt.tight_layout()
    plt.subplots_adjust(left=0.11, bottom=0.08, right=0.95, top=0.94, wspace=0.25, hspace=0.45)
    pass
plt.savefig(traj_clusters_plot_name+".svg", dpi=200)

#plt.show()