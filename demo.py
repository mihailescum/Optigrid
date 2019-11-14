import numpy as np
import matplotlib.pyplot as plt

from optigrid import Optigrid

if __name__ == "__main__":
    # First, generate two separate normal distributions and noise
    normal1_mean = [-5, -5, 1]
    normal1_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 0.05]]
    normal1_samples = 10000
    normal1 = np.random.multivariate_normal(mean=normal1_mean, cov=normal1_cov, size=normal1_samples)

    normal2_mean = [5, 0, -1]
    normal2_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 0.05]]
    normal2_samples = 10000
    normal2 = np.random.multivariate_normal(mean=normal2_mean, cov=normal2_cov, size=normal2_samples)

    noise_low = [-10, -10, -10]
    noise_high = [10, 10, 10]
    noise_samples = 10000
    noise = np.random.uniform(low=noise_low, high=noise_high, size=(noise_samples, 3))

    data = np.concatenate((normal1, normal2))#, noise))

    # Now we want to standard scale our data. Although it is not necessary, it is recommended for better selection of the parameters and uniform importance of the dimensions.
    data_scaled = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Next, chose the parameters
    d = 3 # Number of dimensions
    q = 1 # Number of cutting planes per step
    noise_level = 0.1
    max_cut_score = 0.3

    # Fit Optigrid to the data
    optigrid = Optigrid(d, q, max_cut_score, noise_level, verbose=True)
    optigrid.fit(data_scaled)
    ### Output: 
    ###     In current cluster: 47.08% of datapoints
    ###     In current cluster: 52.92% of datapoints
    ###     Optigrid found 2 clusters.

    for i, cluster in enumerate(optigrid.clusters):
        cluster_data = np.take(data, cluster, axis=0) # Clusters are stored as indices pointing to the original data
        print("Cluster {}: Mean={}, Std={}".format(i, np.mean(cluster_data, axis=0), np.std(cluster_data, axis=0)))
    ### Output: 
    ###     Cluster 0: Mean=[-5.03474967 -3.3355985   0.6569438 ], Std=[1.79700025 4.11403245 3.33377444]
    ###     Cluster 1: Mean=[ 4.92505754  0.05634452 -0.62898176], Std=[1.92237979 3.49116619 3.46671477]

    # Draw a 10 values from both normals and score it with optigrid after normalization
    sample_size = 10
    sample1 = np.random.multivariate_normal(normal1_mean, normal1_cov, sample_size)
    sample2 = np.random.multivariate_normal(normal2_mean, normal2_cov, sample_size)
    sample = np.concatenate((sample1, sample2))
    sample = (sample - np.mean(data)) / np.std(data)

    result = optigrid.score_samples(sample)
    print(result)
    ### Output: 
    ###     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ### The first ten values belong to the zeroth cluster and the latter ten to the second cluster as expected