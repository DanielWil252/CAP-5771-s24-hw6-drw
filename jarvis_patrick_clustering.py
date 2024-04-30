"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.special import comb
import pickle
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
def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    k = params_dict['k']
    smin = params_dict['smin']
 # Step 1: Compute the pairwise Euclidean distances
    distances = cdist(data, data, 'euclidean')
    
    # Step 2: Find k-nearest neighbors (indices) for each point
    knn_indices = np.argsort(distances, axis=1)[:, 1:k+1]
    
    # Step 3: Determine shared neighbors and assign points to clusters
    n_points = data.shape[0]
    clusters = np.full(n_points, -1)
    cluster_id = 0
    for i in range(n_points):
        # Find the points that have at least 'smin' shared neighbors with point 'i'
        shared_neighbors = np.sum(np.isin(knn_indices, knn_indices[i]), axis=1) >= smin
        # Assign the points to the same cluster as point 'i' if they are not already assigned
        for j in np.where(shared_neighbors & (clusters == -1))[0]:
            clusters[j] = cluster_id
        # If point 'i' is not assigned, assign it to a new cluster
        if clusters[i] == -1:
            clusters[i] = cluster_id
            cluster_id += 1
    
    # Compute labels
    computed_labels = clusters

    SSE = sum(np.sum((data[labels == l] - np.mean(data[labels == l], axis=0)) ** 2) for l in np.unique(labels))

    ARI = adjusted_rand_index(labels,computed_labels)



    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """
    data = np.load('question1_cluster_data.npy')
    labels_true = np.load('question1_cluster_labels.npy') 
    answers = {}

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').
    parameter_pairs = [(sigma, xi) for sigma in np.linspace(0.1, 2, 5) for xi in np.linspace(0.1, 2, 2)]
    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    print("Shape of data being passed:", data[:5000].shape)
    
    for i, (sigma, xi) in enumerate(parameter_pairs):
        print("This ran")
        print(f"i = {i}")
        # Conduct spectral clustering on the first 5,000 data points
        labels, SSE, ARI = jarvis_patrick(data[1000*i:1000*(i+1)], labels_true[1000*i:1000*(i+1)], {'sigma': sigma,'k': 5, 'smin': 5})
        # Store results
        groups[i] = {"sigma": sigma,"xi": xi,"ARI": ARI, "SSE": SSE} #removed xi
        if i == 4:
            break
    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}

    # Identify the best and worst parameters based on ARI
    best_params = max(groups.items(), key=lambda x: x[1]['ARI'])
    worst_params = min(groups.items(), key=lambda x: x[1]['ARI'])
    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()

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


    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    ARIs = [best_params[1]['ARI']]  # include ARI from the initial run
    for i in range(1, 5):
        _, _, ARI = jarvis_patrick(data[i * 1000:(i + 1) * 1000], labels_true[i * 1000:(i + 1) * 1000], {'sigma': best_params[1]['sigma'], 'k': 5, 'smin': 5})# May have to do best param for k and smin
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
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
