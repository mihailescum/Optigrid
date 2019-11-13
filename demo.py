import numpy as np
import matplotlib.pyplot as plt

from optigrid import Optigrid

if __name__ == "__main__":
    # First, generate two separate normal distributions and noise
    normal1_mean = [-5, -5]
    normal1_cov = [[1, 0], [0, 1]]
    normal1_samples = 10000
    normal1 = np.random.multivariate_normal(mean=normal1_mean, cov=normal1_cov, size=normal1_samples)

    normal2_mean = [5, 5]
    normal2_cov = [[1, 0], [0, 1]]
    normal2_samples = 10000
    normal2 = np.random.multivariate_normal(mean=normal2_mean, cov=normal2_cov, size=normal2_samples)

    noise_low = [-7, -7]
    noise_high = [7, 7]
    noise_samples = 10000
    noise = np.random.uniform(low=noise_low, high=noise_high, size=(noise_samples,2))

    data = np.concatenate((normal1, normal2, noise))

    # Now we want to standard scale our data. Although it is not necessary, it is recommendet for better selection of the parameters and auniform importance of the dimensions.
    data = (data - np.mean(data)) / np.std(data)


    # Next, chose the parameters
    d = 2 # Number of dimensions
    q = 1 # Number of cutting planes per step
    noise_level = 0.1
    max_cut_score = 0.3

    # Fit Optigrid to the data
    optigrid = Optigrid(d, q, max_cut_score, noise_level, verbose=True)
    optigrid.fit(data)

    # Draw a sample from the second normal and score it with optigrid after normalization
    sample_size = 10
    sample = np.random.multivariate_normal(normal2_mean, normal2_cov, sample_size)
    sample = (sample - np.mean(data)) / np.std(data)

    result = optigrid.score_samples(sample)
    print(result)