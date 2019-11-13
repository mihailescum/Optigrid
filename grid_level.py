class GridLevel:
    def __init__(self, cutting_planes, cluster_index):
        self.cutting_planes = cutting_planes
        self.cluster_index = cluster_index
        self.subgrids = []
        self.subgrid_indices = []

    def add_subgrid(self, subgrid_index, subgrid):
        self.subgrid_indices.append(subgrid_index)
        self.subgrids.append(subgrid)

    def get_sublevel(self, datapoint):
        if datapoint is None:
            raise ValueError("Datapoint must not be None.")

        grid_index = 0
        for i, cut in enumerate(self.cutting_planes):
            if datapoint[cut[1]] > cut[0]:
                grid_index += 2 ** i

        if not grid_index in self.subgrid_indices:
            return None

        return self.subgrids[self.subgrid_indices.index(grid_index)]