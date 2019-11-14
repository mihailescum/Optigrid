class GridLevel:
    """ Optigrid creates a nested partition of the input space. This data structure is used to represent a single level of the grid. Either it represents a cluster or it is devided further into subgrids """

    def __init__(self, cutting_planes, cluster_index):
        """ Creates a grid level.

        Parameters:
            cutting_planes (list): The planes that are used to subdivide this grid level or None if it represents a cluster
            cluster_index (int): The index of the represented cluster or None if it can be subdivided further
        """

        self.cutting_planes = cutting_planes
        self.cluster_index = cluster_index
        self.subgrids = []
        self.subgrid_indices = []

    def add_subgrid(self, subgrid_index, subgrid):
        """ Add a deeper level to the grid

        Parameters:
            subgrid_index (int): For every cutting plane, the subgrid can lay either right or left. This information can be used to binary encode it all at once. This is the subgrid index
            subgrid (GridLevel): The subgrid to add
        """

        self.subgrid_indices.append(subgrid_index)
        self.subgrids.append(subgrid)

    def get_sublevel(self, datapoint):
        """ For a given datapoint returns the subgrid it lies in
        
        Parameters: 
            datapoint (ndarray): The datapoint
        
        Returns:
            GridLevel: The subgrid or None if it belongs to no subgrid, meaning the point is an outlier.
        """

        if datapoint is None:
            raise ValueError("Datapoint must not be None.")

        grid_index = 0
        for i, cut in enumerate(self.cutting_planes):
            if datapoint[cut[1]] > cut[0]:
                grid_index += 2 ** i

        if not grid_index in self.subgrid_indices:
            return None

        return self.subgrids[self.subgrid_indices.index(grid_index)]