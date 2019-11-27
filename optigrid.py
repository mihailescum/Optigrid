import numpy as np
from scipy.stats import gaussian_kde

from grid_level import GridLevel

class Optigrid:
    """ Implementation of the Optigrid Algorithm described in "Optimal Grid-Clustering: Towards Breaking the Curse of Dimensionality in High-Dimensional Clustering" by Hinneburg and Keim """

    def __init__(self, d, q, max_cut_score, noise_level, kde_bandwidth = None, kde_grid_ticks=100, kde_num_samples=15000, kde_atol=1E-6, kde_rtol=1E-4, verbose=False):
        """ 
        Parameters:
            d (int): Dimension of the data
            q (int): Number of cutting planes per iteration
            max_cut_score (double): Maximum density of a cutting plane
        """

        self.d = d
        self.q = q
        self.max_cut_score = max_cut_score
        self.noise_level = noise_level

        self.root = None
        self.clusters = None
        self.num_clusters = -1

        self.kde_bandwidth = kde_bandwidth
        self.kde_grid_ticks = kde_grid_ticks
        self.kde_num_samples = kde_num_samples
        self.kde_atol = kde_atol
        self.kde_rtol = kde_rtol

        self.verbose = verbose

    def fit(self, data, weights=None):
        """ Find all clusters in the data. Clusters are stored as indices pointing to the passed data, i.e. if '10' is in cluster '0' means, that data[10] is in cluster 0.

        Parameters:
            data (ndarray): Each datapoint has to be an array of d dimensions
        """

        data_count = len(data)
        cluster_indices = np.array(range(data_count))

        grid, clusters = self._iteration(data=data, weights=weights, cluster_indices=cluster_indices, percentage_of_values=1, last_cluster_name = [-1])
        self.root = grid
        self.clusters = clusters
        self.num_clusters = len(clusters)

        if self.verbose:
            print("Optigrid found {} clusters.".format(self.num_clusters))

    def _iteration(self, data, weights, cluster_indices, percentage_of_values, last_cluster_name):
        """ Do one recursive step of the optigrid algorithm.

        Parameters:
            data (ndarray): Each datapoint has to be an array of d dimensions
            cluster_indices (list of int): All indices that belong to the current cluster
            percentage_of_values (double): Percentage of values that lay in the current cluster (0-1)
            current_cluster (int): (passed as list to be mutable) The last cluster name that was found, -1 if none

        Returns:
            GridLevel: The gridlevel at the current step with all its depth
            list of list of int: All clusters in the current data chunk
        """

        cuts_iteration = []
        for i in range(self.d): # First create all best cuts
            cuts_iteration += self._create_cuts_kde(data, cluster_indices, current_dimension=i, percentage_of_values=percentage_of_values, weights=weights)
        
        if not cuts_iteration:
            last_cluster_name[0] += 1
            if self.verbose:
                print("Found cluster {}".format(last_cluster_name[0]))

            return GridLevel(cutting_planes=None, cluster_index=last_cluster_name[0]), [cluster_indices]
    
        cuts_iteration = sorted(cuts_iteration, key=lambda x: x[2])[:self.q] # Sort the cuts based on the density at the minima and select the q best ones
        grid = GridLevel(cutting_planes=cuts_iteration, cluster_index=None)
        
        grid_data = self._fill_grid(data, cluster_indices, cuts_iteration) # Fill the subgrid based on the cuts
    
        result = []
        for i, cluster in enumerate(grid_data):
            if cluster.size==0:
                continue
            if self.verbose:
                print("In current cluster: {:.2f}% of datapoints".format(percentage_of_values*len(cluster)/len(cluster_indices)*100))
            subgrid, subresult = self._iteration(data=data, weights=weights, cluster_indices=cluster, percentage_of_values=percentage_of_values*len(cluster)/len(cluster_indices), last_cluster_name=last_cluster_name) # Run Optigrid on every subgrid
            grid.add_subgrid(i, subgrid)
            result += subresult

        return grid, result

    def _fill_grid(self, data, cluster_indices, cuts):
        """ Partitions the grid based on the selected cuts and assignes each cell the corresponding data points (as indices).
        
        Parameters:
            data (ndarray): Each datapoint has to be an array of d dimensions
            cluster_indices (list of int): All indices that belong to the current cluster
            cuts (list): Cutting planes in the format (position, dimension, cutting_score)

        Returns:
            list of list of int: 2**num_cuts lists of indices representing the clusters in this level
        """
        
        num_cuts = len(cuts)
        grid_index = np.zeros(len(cluster_indices))
        for i, cut in enumerate(cuts):
            cut_val = 2 ** i
            grid_index[np.take(np.take(data, cut[1], axis=1), cluster_indices) > cut[0]] += cut_val

        return [cluster_indices[grid_index==key] for key in range(2**num_cuts)]
    
    def _create_cuts_kde(self, data, cluster_indices, current_dimension, percentage_of_values, weights):
        """ Find the best cuts in the specified dimension by estimating the data density using kde.

        Parameters:
            data (ndarray): Each datapoint has to be an array of d dimensions
            cluster_indices (list of int): All indices that belong to the current cluster
            current_dimension (int): Dimension on which to project
            percentage_of_values (double): Percentage of values that lay in the current cluster (0-1)

        Returns:
            list: q best cuts in the format (position, dimension, cutting_score)
        """

        grid, kde = self._estimate_distribution(data, cluster_indices, current_dimension, percentage_of_values=percentage_of_values, weights=weights) 
        kde = np.append(kde, 0)

        peaks = self._find_peaks_distribution(kde)      
        if not peaks:
            return []

        peaks = [peaks[0]] + sorted(sorted(peaks[1:-1], key=lambda x: kde[x], reverse=True)[:self.q - 1]) + [peaks[len(peaks) - 1]] # and get the q-1 most important peaks between the leftest and rightest one.
        best_cuts = self._find_best_cuts(grid, kde, peaks, current_dimension)
        return best_cuts

    def _find_best_cuts(self, grid, kde, peaks, current_dimension):
        """ Using a density estimate and its maxima, finds the best cutting planes
        
        Parameters:
            grid (list of double): The grid on which the density estimate was evaluated
            kde (list of double): For each point on the grid the corresponding density
            peaks (list of double): The maxima of the density estimate on the grid
            current_dimension (int): Dimension on which the data is projected

        Returns:
            list: Best cutting planes in this dimension in the format (position, dimension, cutting_score)
        """
        best_cuts = [] 
        for i in range(len(peaks)-1): # between these peaks search for the optimal cutting plane
            current_min = 1
            current_min_index = -1
            for j in range(peaks[i]+1, peaks[i+1]):
                if kde[j] < current_min:
                    current_min = kde[j]
                    current_min_index = j
            
            if current_min_index >= 0 and current_min < self.max_cut_score:
                best_cuts.append((grid[current_min_index], current_dimension, current_min)) # cutting plane format: (cutting coordinate, dimension in which we cut, density at minimum)
        return best_cuts

    def _find_peaks_distribution(self, kde):
        """ Given a density distribution, locates its peaks

        Parameters:
            kde (list of double): The density estimates on an arbitrary 1D grid

        Returns:
            list of int: The corresponding indices of the grid where the kde has its peaks.
        """

        peaks=[]
        prev = 0
        current = kde[0]
        for bin in range(1, len(kde)): # Find all peaks that are above the noise level
            next = kde[bin] 
            if current > prev and current > next and current >= self.noise_level:
                peaks.append(bin-1)
            prev = current
            current = next
        return peaks

    def _estimate_distribution(self, data, cluster_indices, current_dimension, percentage_of_values, weights):
        """ Estimate the distribution using a sample of the data projected to a coordinate axis using scikits kde estimate method

        Parametes:
            data (ndarray): Each datapoint has to be an array of d dimensions
            cluster_indices (list of int): All indices that belong to the current cluster
            current_dimension (int): Dimension on which to project
            percentage_of_values (double): Percentage of values that lay in the current cluster (0-1)

        Returns:
            list of double: A equally spaced grid
            list of double: The density on the grid points
        """

        sample_size = min(self.kde_num_samples, len(cluster_indices))
        sample = np.random.choice(cluster_indices, size=sample_size)
        datapoints = data[sample][:,current_dimension]
        weights_sample = None
        if not weights is None:
            weights_sample = weights[sample]
        min_val = np.amin(datapoints)
        max_val = np.amax(datapoints)

        std = datapoints.std(ddof=1)
        if np.isclose(std, 0):
            return 0, np.infty

        kde = gaussian_kde(dataset=datapoints, bw_method=self.kde_bandwidth / std, weights=weights_sample)

        grid = np.linspace(min_val, max_val, self.kde_grid_ticks)
        dens = kde.evaluate(grid)
        return grid, dens * percentage_of_values

    def score_samples(self, samples):
        """ For every sample calculates the cluster it belongs to

        Parameters:
            samples (list of ndarray): The sample to score. They need to have the same dimensionality and scale as the data optigrid was fitted with
        
        Returns:
            list of int: For every sample, the cluster it belongs to or None if it is in no cluster (only possible for q>1)
        """

        return [self._score_sample(sample) for sample in samples]

    def _score_sample(self, sample):
        """ Score a single sample

        Parameters:
            sample (ndarray): Needs to have the same dimensionality and scale as the data optigrid was fitted with

        Returns:
            int: Cluster the sample belongs to ore None
        """

        if self.root is None:
            raise Exception("Optigrid needs to be fitted to a dataset first.")

        current_grid_level = self.root
        while current_grid_level.cluster_index is None:
            sub_level = current_grid_level.get_sublevel(sample)
            if sub_level is None:
                return None
            
            current_grid_level = sub_level

        return current_grid_level.cluster_index