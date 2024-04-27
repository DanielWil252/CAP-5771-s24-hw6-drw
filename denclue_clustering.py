"""
Work with DENCLUE clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def denclue(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the DENCLUE algorithm only using the `numpy` module

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'xi'. There could be others.
       params_dict['sigma'] should be in the range [.1, 10], while
       params_dict['xi'] should be in the range .1 to 10. The
       dictionary values scalar float values.

    Return value:
    """
    sigma = params_dict.get('sigma')
    xi = params_dict.get('xi')
    
    if sigma is None or xi is None:
        return None, None, None

    def gaussian_kernel(distance, sigma):
        return np.exp(-0.5 * (distance ** 2) / (sigma ** 2))

    def density_gradient(x, X, sigma):
        differences = X - x
        distances = np.linalg.norm(differences, axis=1)
        weights = gaussian_kernel(distances, sigma)
        return np.sum(differences * weights[:, np.newaxis], axis=0) / (sigma ** 2)

    def find_local_maxima(x, X, sigma, xi, max_iters=100, tol=1e-5):
        for _ in range(max_iters):
            grad = density_gradient(x, X, sigma)
            x_new = x + xi * grad
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        return x

    n_points = data.shape[0]
    cluster_centers = []
    visited = np.zeros(n_points, dtype=bool)
    computed_labels = -np.ones(n_points, dtype=np.int32)

    # Find local maxima and identify clusters
    for i in range(n_points):
        if not visited[i]:
            visited[i] = True
            local_max = find_local_maxima(data[i], data, sigma, xi)
            is_unique = True
            for center in cluster_centers:
                if np.linalg.norm(local_max - center) < xi:
                    is_unique = False
                    computed_labels[i] = cluster_centers.index(center)
                    break
            if is_unique:
                cluster_centers.append(local_max)
                computed_labels[i] = len(cluster_centers) - 1

    # Compute SSE for clusters
    SSE = 0.0
    if len(cluster_centers) > 0:
        for k in range(len(cluster_centers)):
            cluster_data = data[computed_labels == k]
            cluster_center = cluster_centers[k]
            SSE += np.sum((cluster_data - cluster_center) ** 2)

    return computed_labels, SSE, None  # ARI is not computed, so return None
def load_data():
    # Load the data and labels from the .npy files
    data = np.load('cluster_data.npy')
    labels = np.load('cluster_labels.npy')
    return data, labels

def plot_data(data):
    plt.scatter(data[:, 0], data[:, 1], s=1)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Original Data Scatterplot')
    plt.grid(True)
    plt.show()

def hyperparameter_study(data, sigmas, xis):
    results = []
    for sigma in sigmas:
        for xi in xis:
            computed_labels, SSE, _ = denclue(data, sigma, xi)
            # ARI would be computed here if labels were provided
            results.append({'sigma': sigma, 'xi': xi, 'SSE': SSE})
    return results

def denclue_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.

    """

    answers = {}

    # Return your `denclue` function
    answers["denclue_function"] = denclue

    data, labels = load_data()
    plot_data(data[0:10000])
    plot_cluster = plt.scatter([1,2,3], [4,5,6])
    answers["plot_original_cluster"] = plot_cluster

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using DENCLUE
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.  For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # Variable `groups` is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    """
    plt.scatter(.....)
    plt.xlabel(....)
    plt.ylabel(...)
    plt.title(...)
    plot_ARI = plt
    """

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter([1,2,3], [4,5,6])
    plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["scatterplot cluster with largest ARI"] = plot_ARI
    answers["scatterplot cluster with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = 0.

    # A single float
    answers["std_ARIs"] = 0.

    # A single float
    answers["mean_SSEs"] = 0.

    # A single float
    answers["std_SSEs"] = 0.

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = denclue_clustering()
    with open("denclue_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)

