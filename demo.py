import numpy as np

from optigrid import Optigrid

if __name__ == "__main__":
    # First, generate two separate normal distributions
    normal1_mean = [-5]
    normal1_cov = [[1]]
    normal1_samples = 10000
    normal1 = np.random.multivariate_normal(normal1_mean, normal1_cov, normal1_samples)

    normal2_mean = [5]
    normal2_cov = [[1]]
    normal2_samples = 10000
    normal2 = np.random.multivariate_normal(normal2_mean, normal2_cov, normal2_samples)

    data = np.concatenate((normal1, normal2))

    # Now we want to standard scale our data. Although it is not necessary, it is recommendet for better selection of the parameters and auniform importance of the dimensions.
    data = (data - np.mean(data)) / np.std(data)

    # Next, chose the parameters
    d = 1 # Number of dimensions
    q = 1 # Number of cuttingplanes per step
    noise_level = 0.1
    max_cut_score = 0.3

    optigrid = Optigrid(d, q, max_cut_score, noise_level, verbose=True)
    optigrid.fit(data)