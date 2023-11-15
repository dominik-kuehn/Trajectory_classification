#standard trajectory classification code with layout to run over different values of a parameter of the
#process, or different quantities to be clustered, to optimise the workflow

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
        'size': 18,
        }

plt.rcParams.update({'font.size': 14})

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
n = 40000 #Number of particles to plot

#first electrons exiting after approx 280 cycles, ions are slow, we can use all the cycles
cycles = 300

#mode for adaptive run
mode = 15

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
x = x_original - M[0,0:nx:3]            #centralise trajectories
x = x/np.max(np.absolute(x), axis=0)    #normalise
x = np.absolute(np.fft.fft(x, axis=0))  #fft and remove negative frequencies
x = x/np.max(x, axis=0)                 #renormalise

# Y - FFT
y = y_original - M[0,1:ny:3]
y = y/np.max(np.absolute(y), axis=0)
y = np.absolute(np.fft.fft(y, axis=0))
y = y/np.max(y, axis=0)

#energy
energy = energy_original #- M[0,2:ne:3]
energy = energy/np.max(np.absolute(energy), axis=0)
#energy = np.absolute(np.fft.fft(energy, axis=0))   #energy fft test runs
#energy = energy/np.max(energy, axis=0)
energy_average = np.mean(energy, axis=0)


'''
#FFT plots
plt.figure()
plt.plot(np.fft.fftfreq(300), x[:,10])
#plt.plot(x[:,0]) #np.fft.fftfreq(300), 
plt.title('FFT of x', fontdict=title_font)
plt.xlabel('$\omega/\omega_p$')
plt.ylabel('A')

fig, ax = plt.subplots()
ax.plot(x[:150,:10])
ax2 = ax.twiny()
ax2.plot(4*np.fft.fftfreq(300)[:150], x[:150,:10])
ax.vlines(20, 0, 1, color= "k", linestyles="dotted", label="Cut-Off")
ax.tick_params(axis="x", colors="C0")
ax2.tick_params(axis="x", colors="C1")
ax.set_xlabel('Fourier modes', color="C0")
ax.set_ylabel('Rel. Amplitude')
ax2.set_xlabel('Frequency $\omega/\omega_p$', color="C1")
fig.subplots_adjust(top=0.802, bottom=0.124)
ax.set_title("FFT on x", fontdict=title_font)
ax.legend(loc="upper right")
fig.savefig("FFT_X_selected.png", dpi=200)

fig, ax = plt.subplots()
ax.plot(y[:150,:10])
ax2 = ax.twiny()
ax2.plot(4*np.fft.fftfreq(300)[:150], y[:150,:10])
ax.vlines(20, 0, 1, color= "k", linestyles="dotted", label="Cut-Off")
ax.tick_params(axis="x", colors="C0")
ax2.tick_params(axis="x", colors="C1")
ax.set_xlabel('Fourier modes', color="C0")
ax.set_ylabel('Rel. Amplitude')
ax2.set_xlabel('Frequency $\omega/\omega_p$', color="C1")
fig.subplots_adjust(top=0.802, bottom=0.124)
ax.set_title("FFT on y", fontdict=title_font)
ax.legend(loc="upper right")
fig.savefig("FFT_Y_selected.png", dpi=200)

plt.show()
'''


for mode in [2,4,6,8,12,16,20,24]:#[1,2,5,10,15,20,25,30,50,100,120,150,200,300]:
    plt.close()
    print("Current mode:", mode)
    #PCA for dimensionality reduction
    num_spectral_modes = 300#int(cycles/2) # can try just with a part of spectrum: cycles/4, cycles/8

    traj =np.vstack((x[:num_spectral_modes,:], y[:num_spectral_modes,:])) #, energy_average))
    #traj =np.vstack((energy)) #[:energy_modes,:]))
    #traj =np.vstack((x[:num_spectral_modes,:], y[:num_spectral_modes,:], energy[:energy_modes,:]))
    pca = PCA()  #(.95) means sk chooses n s.t. 95% of variance is preserved #(n_components=20)
    pca.fit(traj.T)

    coefs = pca.components_.T
    score = pca.transform(traj.T)
    explained = pca.explained_variance_ratio_



    #plot_file_names
    traj_clusters_plot_name = "traj_clusters_plot_en_fft_20"#+str(mode)
    en_evolution_clusters_plot_name = "en_evolution_clusters_plot_en_fft_20"#+str(mode)
    en_distribution_clusters_plot_name = "en_distribution_clusters_plot_en_fft_20"#+str(mode)
    pca_cluster_plot_name = "pca_clusters_plot_stand_en_fft_20"#+str(mode)
    pareto_plot_name = "pareto_plot_stand_en_fft_20"#+str(mode)


    # choose the reduced number of PCA components to use in the analysis
    num_pca_components = mode
    #if num_pca_components > 2*mode:
    #     num_pca_components = 2*mode

    #pca_expl_arr = []
    for i in range(num_pca_components):
        pca_expl = sum(explained[0:i+1])
        #pca_expl_arr.append(pca_expl)
        print(pca_expl)
    pca_expl = sum(explained[:num_pca_components])
    print("Total Variance:", pca_expl)
    #Clustering
    print('Clustering ...')

    #for reproducibility
    np.random.seed(12)



    # k-means -scikit
    num_clusters = 12 
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=50, max_iter=1000, tol=1e-4, algorithm='lloyd')
    idx = kmeans.fit_predict(score[:,:num_pca_components], sample_weight=None)
    C = kmeans.cluster_centers_

    
    #plot pca clustered data 
    plt.figure()
    plt.scatter(score[:,0], score[:,1], num_clusters, idx)
    plt.title('K-means -Euclidean Distance', fontdict=title_font)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.subplots_adjust(left=0.16, right=0.92)
    plt.savefig(pca_cluster_plot_name, dpi=150)

    #plot pareto graph
    dim = np.arange(1,num_pca_components+1) #the x-dimension in this plot
    fig, ax = plt.subplots()
    ax.bar(dim, 100*explained[:num_pca_components], color="C0")
    ax2 = ax.twinx()
    ax2.plot(dim, [sum(explained[:i])*100 for i in dim], color="C1", marker="D", ms=7)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim([0.3,num_pca_components+0.7])
    ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C1")
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('% of total variance per PC', color="C0")
    ax2.set_ylabel('Accumulated % of total variance', color="C1")
    fig.subplots_adjust(right=0.857)
    fig.savefig(pareto_plot_name, dpi=150)
    plt.close()
    
    #count the number of clusters per idx
    #plt.figure()
    counts,_,_ = plt.hist(idx, num_clusters)
    #plt.xlabel('Cluster ID')
    #plt.ylabel('#Samples')
    #plt.grid()
    out = np.sort(counts)[::-1]
    ii = np.argsort(counts)[::-1]
    perc = counts[ii]/n*100

    print('Calculating Silhouette...')
    silh2 = silhouette(score[:,:num_pca_components], idx)
    bad_samples = np.sum(silh2<0)

    print("Bad samples:", bad_samples)


    #plot typical traj for different clusters

    plotted_particles=25
    fig_tr, ax_tr = plt.subplots(4, 3, figsize=(25,11))
    fig_en, ax_en = plt.subplots(4, 3, figsize=(25,11))
    fig_ev, ax_ev = plt.subplots(4, 3, figsize=(25,11))

    ax_tr = np.reshape(ax_tr, (12,1)) #for easy access to correct subplots using ax_tr[i][0]
    ax_en = np.reshape(ax_en, (12,1))
    ax_ev = np.reshape(ax_ev, (12,1))

    print("Plotting classes")
    for i in range(num_clusters):
        print("population of",i,":", out[i])
        silh_i, x_i, y_i, e_i  = silh2[idx==ii[i]], x_original[:, idx==ii[i]], y_original[:, idx==ii[i]], energy_original[:, idx==ii[i]]  #selecting sil-score, x, y, e of right class
        silh_topindex = np.array(np.argsort(silh_i))[::-1]  #sorting indices of silhouette scores in descending order
        xi = x_i[:, silh_topindex[:plotted_particles]]  #x positions
        yi = y_i[:, silh_topindex[:plotted_particles]]  #y positions
        e_final_i = e_i[-1, silh_topindex[:int(out[i])]]#final energy distributions
        ei = e_i[:, silh_topindex[:plotted_particles]]  #energy evolution in time

        ax_tr[i][0].plot(xi, yi)
        ax_tr[i][0].set_title("%d: %3.1f"%(i+1, perc[i]), fontdict=title_font)
        ax_tr[i][0].set_xlabel("x")
        ax_tr[i][0].set_ylabel("y")
        ax_tr[i][0].grid()
        ax_tr[i][0].axis([12,28,8,12])

        ax_en[i][0].hist(e_final_i, bins = int(np.sqrt(out[i])))
        ax_en[i][0].set_title("%d: %3.1f"%(i+1, perc[i]), fontdict=title_font)
        ax_en[i][0].set_xlabel("E")
        ax_en[i][0].set_ylabel("#")
        ax_en[i][0].grid()
        ax_en[i][0].set_xlim([0,0.6])

        ax_ev[i][0].plot(ei)
        ax_ev[i][0].set_title("%d: %3.1f"%(i+1, perc[i]), fontdict=title_font)
        ax_ev[i][0].set_xlabel("time step")
        ax_ev[i][0].set_ylabel("E")
        ax_ev[i][0].grid()
        #ax_ev[i][0].axis([12,28,8,12])
        print("class", i, "done")
        pass
    
    fig_tr.subplots_adjust(left=0.05, bottom=0.07, right=0.95, top=0.94, wspace=0.25, hspace=0.56) 
    fig_tr.savefig(traj_clusters_plot_name, dpi=200)
    fig_en.subplots_adjust(left=0.05, bottom=0.07, right=0.95, top=0.94, wspace=0.25, hspace=0.56) 
    fig_en.savefig(en_distribution_clusters_plot_name, dpi=200)
    fig_ev.subplots_adjust(left=0.05, bottom=0.07, right=0.95, top=0.94, wspace=0.25, hspace=0.56) 
    fig_ev.savefig(en_evolution_clusters_plot_name, dpi=200)
    plt.close()