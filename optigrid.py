import pandas as pd
import numpy as np
import random
from sklearn.neighbors import KernelDensity

from grid_level import GridLevel

class Optigrid:
    def __init__(self, d, q, max_cut_score, noise_level, kde_kernel='gaussian', kde_bandwidth = 0.1, kde_grid_ticks=100, kde_num_samples=15000, kde_atol=1E-6, kde_rtol=1E-4, verbose=False):
        self.d = d
        self.q = q
        self.max_cut_score = max_cut_score
        self.noise_level = noise_level

        self.root = None
        self.cluster = None
        self.num_clusters = -1

        self.kde_kernel = kde_kernel
        self.kde_bandwidth = kde_bandwidth
        self.kde_grid_ticks = kde_grid_ticks
        self.kde_num_samples = kde_num_samples
        self.kde_atol = kde_atol
        self.kde_rtol = kde_rtol

        self.verbose = verbose

    def fit(self, data):
        data_count = len(data)
        cluster_indices = np.array(range(data_count))

        grid, clusters = self._iteration(data=data, cluster_indices=cluster_indices, percentage_of_values=1, current_cluster = [-1])
        self.root = grid
        self.clusters = clusters
        self.num_clusters = len(clusters)

        if self.verbose:
            print("Optigrid found {} clusters.".format(self.num_clusters))

    def _iteration(self, data, cluster_indices, percentage_of_values, current_cluster):
        cuts_iteration = []
        for i in range(self.d): # First create all best cuts
            cuts_iteration += self._create_cuts_kde(data, cluster_indices, current_dimension=i, percentage_of_values=percentage_of_values)
        
        if not cuts_iteration:
            current_cluster[0] += 1
            return GridLevel(cutting_planes=None, cluster_index=current_cluster[0]), [cluster_indices]
    
        cuts_iteration = sorted(cuts_iteration, key=lambda x: x[2])[:self.q] # Sort the cuts based on the density at the minima and select the q best ones
        grid = GridLevel(cutting_planes=cuts_iteration, cluster_index=None)
        
        grid_data = self._fill_grid(data, cluster_indices, cuts_iteration) # Fill the subgrid based on the cuts
    
        result = []
        for i, cluster in enumerate(grid_data):
            if cluster.size==0:
                continue
            if self.verbose:
                print("In current cluster: {:.2f}% of datapoints".format(percentage_of_values*len(cluster)/len(cluster_indices)*100))
            subgrid, subresult = self._iteration(data=data, cluster_indices=cluster, percentage_of_values=percentage_of_values*len(cluster)/len(cluster_indices), current_cluster=current_cluster) # Run Optigrid on every subgrid
            grid.add_subgrid(i, subgrid)
            result.append(subresult)

        return grid, result

    def _fill_grid(self, data, cluster_indices, cuts):
        """ Partitions the grid based on the selected cuts and assignes each cell the corresponding data points (as indices)"""
        
        num_cuts = len(cuts)
        grid_index = np.zeros(len(cluster_indices))
        for i, cut in enumerate(cuts):
            cut_val = 2 ** i
            grid_index[np.take(np.take(data, cut[1], axis=1), cluster_indices) > cut[0]] += cut_val

        return [cluster_indices[grid_index==key] for key in range(2**num_cuts)]
    
    def _create_cuts_kde(self, data, cluster_indices, current_dimension, percentage_of_values):
        grid, kde = self._estimate_distribution(data, cluster_indices, current_dimension, percentage_of_values=percentage_of_values) 
        kde = np.append(kde, 0)

        peaks = self._find_peaks_distribution(kde)      
        if not peaks:
            return []

        peaks = [peaks[0]] + sorted(sorted(peaks[1:-1], key=lambda x: kde[x], reverse=True)[:self.q - 1]) + [peaks[len(peaks) - 1]] # and get the q-1 most important peaks between the leftest and rightest one.
        best_cuts = self._find_best_cuts(grid, kde, peaks, current_dimension)
        return best_cuts

    def _find_best_cuts(self, grid, kde, peaks, current_dimension):
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

    def _estimate_distribution(self, data, cluster_indices, current_dimension, percentage_of_values):
        sample_size = min(self.kde_num_samples, len(cluster_indices))
        sample = np.random.choice(cluster_indices, size=sample_size)
        datapoints = np.expand_dims(data[sample][:,current_dimension], -1)
        min_val = np.amin(datapoints)
        max_val = np.amax(datapoints)

        kde = KernelDensity(kernel=self.kde_kernel, bandwidth=self.kde_bandwidth, atol=self.kde_atol, rtol=self.kde_rtol).fit(datapoints)

        grid = np.linspace([min_val], [max_val], self.kde_grid_ticks)
        log_dens = kde.score_samples(grid)
        return grid, np.exp(log_dens) * percentage_of_values

    def score_samples(self, samples):
        return [self._score_sample(sample) for sample in samples]

    def _score_sample(self, sample):
        if self.root is None:
            raise Exception("Optigrid needs to be fitted to a dataset first.")

        current_grid_level = self.root
        while current_grid_level.cluster_index is None:
            sub_level = current_grid_level.get_sublevel(sample)
            if sub_level is None:
                return None
            
            current_grid_level = sub_level

        return current_grid_level.cluster_index