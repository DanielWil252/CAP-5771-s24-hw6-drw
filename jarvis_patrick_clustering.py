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

def adjusted_rand_index(labels_true, labels_pred) -> float:
    """
    Compute the adjusted Rand index.

    Parameters:
    - labels_true: The true labels of the data points.
    - labels_pred: The predicted labels of the data points.

    Returns:
    - ari: The adjusted Rand index value.
    """
    # Create contingency table
    contingency_table = np.histogram2d(
        labels_true, labels_pred, 
        bins=(np.unique(labels_true).size, np.unique(labels_pred).size)
    )[0]

    # Sum over rows and columns
    sum_combinations_rows = np.sum([np.sum(row) * (np.sum(row) - 1) / 2 for row in contingency_table])
    sum_combinations_cols = np.sum([np.sum(col) * (np.sum(col) - 1) / 2 for col in contingency_table.T])

    # Sum of combinations for all elements
    N = np.sum(contingency_table)
    sum_combinations_total = N * (N - 1) / 2

    # Calculate ARI
    sum_combinations_within = np.sum([n * (n - 1) / 2 for n in contingency_table.flatten()])

    ari = (
        sum_combinations_within - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    ) / (
        (sum_combinations_rows + sum_combinations_cols) / 2
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    )
    return ari
def compute_SSE(data, labels):
    """
    Calculate the sum of squared errors (SSE) for a clustering.

    Parameters:
    - data: numpy array of shape (n, 2) containing the data points
    - labels: numpy array of shape (n,) containing the cluster assignments

    Returns:
    - sse: the sum of squared errors
    """
    # ADD STUDENT CODE
    sse = 0.0
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    return sse
# SSE calculation
def sse(data, labels):
    if len(data) != len(labels):
        raise ValueError("The length of data and predictions must be the same.")
    sse = sum((d - p) ** 2 for d, p in zip(data, labels))
    return sse

def jarvis_patrick(
    data: NDArray[np.float64], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - labels: true labels of the dataset for ARI calculation.
    - params_dict: dictionary of parameters. Must include 'k' and 'smin'.
    - params_dict['k']: the number of nearest neighbors to consider.
    - params_dict['smin']: the minimum number of shared neighbors for clustering.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: sum of squared errors
    - ARI: adjusted Rand index
    """

    k = params_dict['k']
    smin = params_dict['smin']
    
    # Calculate pairwise Euclidean distance
    dist_matrix = np.sqrt(((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Get indices of k nearest neighbors for each point
    neighbors = np.argsort(dist_matrix, axis=1)[:, 1:k+1]

    # Initialize cluster labels
    cluster_labels = -np.ones(data.shape[0], dtype=np.int32)
    current_cluster = 0
    
    for i in range(data.shape[0]):
        if cluster_labels[i] == -1:  # If point i is not yet labeled
            # Start a new cluster with point i
            cluster_labels[i] = current_cluster
            
            # Find all points with enough shared neighbors with point i
            for j in range(data.shape[0]):
                if i != j and cluster_labels[j] == -1:
                    shared_neighbors = np.intersect1d(neighbors[i], neighbors[j], assume_unique=True)
                    if len(shared_neighbors) >= smin:
                        cluster_labels[j] = current_cluster
            
            current_cluster += 1

    # Compute SSE
    # sse = sum(np.sum((data[labels == l] - np.mean(data[labels == l], axis=0)) ** 2) for l in np.unique(labels)) 
    sse = compute_SSE(data,labels)
    # Compute ARI
    ari = adjusted_rand_index(labels, cluster_labels)

    return cluster_labels, sse, ari


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
    parameter_pairs = [(k, smin) for k in [3,4,5,6,7] for smin in [4,6,8,10]]
    # print(parameter_pairs)
    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    # print("Shape of data being passed:", data[:5000].shape)
    
    for i, (k, smin) in enumerate(parameter_pairs):
        # print(f"i = {i}")
        # Conduct spectral clustering on the first 5,000 data points
        labels, SSE, ARI = jarvis_patrick(data[1000*i:1000*(i+1)], labels_true[1000*i:1000*(i+1)], {'k': k, 'smin': smin})
        # Store results
        groups[i] = {"k": k,"ARI": ARI, "SSE": SSE,"smin": smin} #removed xi
        #if i == 4:
        #    break
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
    # print(best_params)
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
    smins = [group["smin"] for group in groups.values()]
    ks = [group["k"] for group in groups.values()]
    # xis = [group["xi"] for group in groups.values()]
    SSEs = [group["SSE"] for group in groups.values()]
    ARIs = [group["ARI"] for group in groups.values()]
    plt.figure()
    #plt.scatter(sigmas, xis, c=SSEs)
    plot_SSE = plt.scatter(ks,smins,c=SSEs)
    plt.colorbar(label='SSE')
    plt.xlabel('K')
    plt.ylabel('Smin')
    plt.title('Scatter Plot Colored by SSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("JV_clustering_SSE.png")
    # plt.show()
    # print(f"SSEs: {SSEs}")
    plt.figure()
    plot_ARI = plt.scatter(ks, smins, c=ARIs)
    plt.colorbar(label='ARI')
    plt.xlabel('K')
    plt.ylabel('Smin')
    plt.title('Scatter Plot Colored by ARI')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("JV_clustering_ARI.png")
    # plt.show()    
    # print(f"ARIs: {ARIs}")

    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    ARIs = [best_params[1]['ARI']]  # include ARI from the initial run
    for i in range(1, 5):
        _, _, ARI = jarvis_patrick(data[i * 1000:(i + 1) * 1000], labels_true[i * 1000:(i + 1) * 1000], {'k': best_params[1]["k"], 'smin': best_params[1]["smin"]})# May have to do best param for k and smin
        ARIs.append(ARI)
    
    # Calculate mean and standard deviation for ARI and SSE
    answers["mean_ARIs"] = np.mean(ARIs)
    answers["std_ARIs"] = np.std(ARIs)
    answers["mean_SSEs"] = np.mean(SSEs)
    answers["std_SSEs"] = np.std(SSEs)
    # print(np.mean(ARIs))
    # print(np.std(ARIs)) 

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
