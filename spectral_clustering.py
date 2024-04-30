"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigs
from scipy.cluster.vq import kmeans2
from scipy.linalg import eigh
from scipy.special import comb
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.vq import kmeans2
import pickle
#from sklearn.cluster import KMeans
######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def adjusted_rand_index(labels_true, labels_pred):
   # Create a contingency table from labels
    contingency = np.histogram2d(labels_true, labels_pred, bins=(np.unique(labels_true).size, np.unique(labels_pred).size))[0]

    # Sum over rows & columns
    sum_rows = np.sum(contingency, axis=1)
    sum_cols = np.sum(contingency, axis=0)

    # Total combinations of pairs
    total_combinations = comb(contingency.sum(), 2)

    # Combinations within the same clusters
    comb_within_clusters = np.sum([comb(n, 2) for n in contingency.flatten()])
    comb_same_cluster_true = np.sum([comb(n, 2) for n in sum_rows])
    comb_same_cluster_pred = np.sum([comb(n, 2) for n in sum_cols])

    # Expected index
    expected_index = (comb_same_cluster_true * comb_same_cluster_pred) / total_combinations

    # Max index
    max_index = (comb_same_cluster_true + comb_same_cluster_pred) / 2

    # Adjusted Rand Index
    ari = (comb_within_clusters - expected_index) / (max_index - expected_index)

    return ari 
# SSE calculation
def sse(data, labels):
    if len(data) != len(labels):
        raise ValueError("The length of data and predictions must be the same.")
    sse = sum((d - p) ** 2 for d, p in zip(data, labels))
    return sse


def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    sigma = params_dict['sigma']
    k = params_dict['k']

    # Step 1: Create a sparsified similarity graph G

    # Compute the RBF (Gaussian) kernel similarity
    pairwise_sq_dists = squareform(pdist(data, 'sqeuclidean'))

    affinity_matrix = np.exp(-pairwise_sq_dists / (sigma ** 2))

    # Step 2: Compute the graph Laplacian for G, L
    laplacian = csgraph.laplacian(affinity_matrix, normed=False)
    
    # Step 3: Create a matrix V from the first k eigenvectors of L
    # Since we want the smallest eigenvalues, we use 'SM' which stands for 'smallest magnitude'
    eigenvalues, eigenvectors = eigsh(laplacian, k=k, which='SM')

    # Step 4: Apply K-means clustering on V to obtain the k clusters
    # Use the real part of eigenvectors in case they are complex numbers
    v_matrix = eigenvectors.real

    centroids, computed_labels = kmeans2(v_matrix, 5, minit='random')

    SSE = sum(np.sum((data[labels == l] - np.mean(data[labels == l], axis=0)) ** 2) for l in np.unique(labels))

    #ARI = adjusted_rand_index(labels, computed_labels) 
    ARI = adjusted_rand_index(labels,computed_labels)
    return computed_labels, SSE, ARI, eigenvalues    


def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}
    data = np.load('question1_cluster_data.npy')
    labels_true = np.load('question1_cluster_labels.npy')    
 
    # Return your `spectral` function
    answers["spectral_function"] = spectral

    parameter_pairs = [(sigma, xi) for sigma in np.linspace(0.1, 2, 5) for xi in np.linspace(0.1, 2, 2)]
    print(parameter_pairs)
    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    print("Shape of data being passed:", data[:5000].shape)
    
    for i, (sigma, xi) in enumerate(parameter_pairs):
        print("This ran")
        print(f"i = {i}")
        # Conduct spectral clustering on the first 5,000 data points
        labels, SSE, ARI, eigenvalues = spectral(data[1000*i:1000*(i+1)], labels_true[1000*i:1000*(i+1)], {'sigma': sigma, 'k': 5})
        # Store results
        groups[i] = {"sigma": sigma,"xi": xi,"ARI": ARI, "SSE": SSE} #removed xi
        if i == 0:
            # Save the eigenvalues for plotting
            answers["eigenvalues"] = eigenvalues    
        if i == 4:
            break
    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}
 # Identify the best and worst parameters based on ARI
    best_params = max(groups.items(), key=lambda x: x[1]['ARI'])
    worst_params = min(groups.items(), key=lambda x: x[1]['ARI'])

    # Plot SSE and ARI scatter plots
    sigmas = [group["sigma"] for group in groups.values()]
    xis = [group["xi"] for group in groups.values()]
    SSEs = [group["SSE"] for group in groups.values()]
    ARIs = [group["ARI"] for group in groups.values()]
    plt.figure()
    #plt.scatter(sigmas, xis, c=SSEs)
    plot_SSE = plt.scatter(sigmas,xis,c=SSEs)
    plt.colorbar(label='SSE')
    plt.xlabel('Sigma')
    plt.ylabel('Xi')
    plt.title('Scatter Plot Colored by SSE')
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plot_ARI = plt.scatter(sigmas, xis, c=ARIs)
    plt.colorbar(label='ARI')
    plt.xlabel('Sigma')
    plt.ylabel('Xi')
    plt.title('Scatter Plot Colored by ARI')
    plt.grid(True)
    plt.show()    
    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    #plot_ARI = plt.scatter([1,2,3], [4,5,6])
    #plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.

    # Plot eigenvalues
    plt.figure()
    plot_eig = plt.plot(answers["eigenvalues"])
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues from Smallest to Largest')
    plt.grid(True)
    plt.show()
    answers["eigenvalue plot"] = plot_eig

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    # Apply the best parameters to the rest of the data
    print(best_params[1])
    ARIs = [best_params[1]['ARI']]  # include ARI from the initial run
    for i in range(1, 5):
        _, _, ARI, _ = spectral(data[i * 1000:(i + 1) * 1000], labels_true[i * 1000:(i + 1) * 1000], {'sigma': best_params[1]['sigma'], 'k': 5})
        ARIs.append(ARI)
    
    # Calculate mean and standard deviation for ARI and SSE
    answers["mean_ARIs"] = np.mean(ARIs)
    answers["std_ARIs"] = np.std(ARIs)
    answers["mean_SSEs"] = np.mean(SSEs)
    answers["std_SSEs"] = np.std(SSEs)
    print(np.mean(ARIs))
    print(np.std(ARIs))
    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
