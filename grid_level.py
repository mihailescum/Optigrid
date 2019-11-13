class GridLevel:
    def __init__(self, cutting_planes):
        self.cutting_planes = cutting_planes
        self.subgrids = []
        self.subgrid_indices = []

    def add_subgrid(self, subgrid_index, subgrid):
        self.subgrid_indices.append(subgrid_index)
        self.subgrids.append(subgrid)