import numpy as np
from generate_target import return_numpy_array_for_letter
import umap 
from sklearn.decomposition import PCA
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
import phate
from wasserstein import demo_wasserstein as wasserstein_distance
import matplotlib.pyplot as plt

class UMAP2:
    def __init__(self, 
                 target_string,
                 target_size = 8, 
                 intermediate_dimension = 2, # this is fixed at 2 for now
                 intermediate_granularity = 8,
                 initial_dimensionality_reducer='JL'):
        
        self.target_string = target_string
        self.target_size = target_size
        self.dim_red_method = initial_dimensionality_reducer
        self.intermediate_granularity = intermediate_granularity
        self.intermediate_dimension = intermediate_dimension

        if target_size != intermediate_granularity:
            raise NotImplementedError("At the moment, target_size must equal intermediate_granularity")

        # initialize the target word
        self.target_img = return_numpy_array_for_letter(target_string, target_size)
        
        # create the initial dimensionality reducer (dr1)
        if self.dim_red_method == "PCA":
            self.dr1 = PCA(n_components=self.intermediate_dimension)
        elif self.dim_red_method == "JL":
            self.dr1 = SparseRandomProjection(n_components=self.intermediate_dimension)
        elif self.dim_red_method == "UMAP":
            self.dr1 = umap.UMAP(n_components=self.intermediate_dimension)
        elif self.dim_red_method == "PHATE":
            self.dr1 = phate.PHATE(n_components=self.intermediate_dimension)
        else:
            raise ValueError("Invalid initial_dimensionality_reducer option")


    def fit_transform(self, data):
        # apply the intermediate dimensionality reduction
        print(f'Transforming data with {self.dim_red_method}...')
        intermediate_data = self.dr1.fit_transform(data)
        # TODO: finish function
        print(f'Computing wasserstein distance and transport map...')
        grid_assignments, grid_point_probs, grid_locations, delta = self.discretize_points(intermediate_data)
        dist, T = self.compute_wasserstein(grid_point_probs, grid_locations)
        print(f'Wasserstein distance: {dist}')

        new_grid_assignment, data_loc = self.update_grid_assignment(len(data), grid_assignments, T, grid_locations, delta)
        return data_loc
    
    def compute_wasserstein(self, grid_point_probs, grid_locations):
        # normalize and flatten the image to be a probability distribution
        target_img = self.target_img.flatten()
        target_img = target_img / np.sum(target_img)

        # compute the wasserstein distance between the target image and the grid point probabilities
        dist, T = wasserstein_distance(grid_locations, grid_point_probs, target_img)
        return dist, T

    def discretize_points(self, intermediate_data):
        intermediate_granularity = self.intermediate_granularity

        min_x = np.min(intermediate_data[:,0])
        max_x = np.max(intermediate_data[:,0])
        min_y = np.min(intermediate_data[:,1])
        max_y = np.max(intermediate_data[:,1])
        # normalize data to [0, 1]
        intermediate_data[:,0] = (intermediate_data[:,0] - min_x) / (max_x - min_x)
        intermediate_data[:,1] = (intermediate_data[:,1] - min_y) / (max_y - min_y)


        x_axis_step = 1 / intermediate_granularity
        y_axis_step = 1 / intermediate_granularity

        x_grid_assignment = intermediate_data[:,0] // x_axis_step
        y_grid_assignment = intermediate_granularity - intermediate_data[:,1] // y_axis_step
        
        # if any grid assignment value is equal to intermediate_granularity, subtract 1
        x_grid_assignment[x_grid_assignment == intermediate_granularity] -= 1
        y_grid_assignment[y_grid_assignment == intermediate_granularity] -= 1

        grid_assignments = x_grid_assignment + (intermediate_granularity * y_grid_assignment)

        # get a discrete probability distribution for the grid points
        # make grid_assignment_list positive integers 
        grid_assignment_list = np.abs(grid_assignments).astype(int)
        #grid_assignment_list = np.abs(grid_assignments).astype(int)
        # check if any grid assignments are negative

        if np.any(grid_assignment_list < 0):
            import pdb; pdb.set_trace()
            # print count of negative grid points 
            print(f'Number of negative grid points: {np.sum(grid_assignment_list < 0)}')
        grid_point_counts = np.bincount(grid_assignment_list)
        if len(grid_point_counts) < intermediate_granularity**2:
            grid_point_counts = np.append(grid_point_counts, np.zeros(intermediate_granularity**2 - len(grid_point_counts)))
        grid_point_probs = grid_point_counts / np.sum(grid_point_counts)
                
        # get location of each grid location
        grid_locations = np.zeros((intermediate_granularity**2, 2))
        for i in range(intermediate_granularity**2):
            x_coord = (i % intermediate_granularity) * 1/intermediate_granularity
            y_coord = 1 - (i // intermediate_granularity) * 1/intermediate_granularity
            grid_locations[i] = [x_coord, y_coord]

        # get the vector between each intermediate data point and its assigned grid point
        delta = np.zeros((len(intermediate_data), 2))
        for i in range(len(intermediate_data)):
            grid_assignment = grid_assignments[i]
            grid_location = grid_locations[int(grid_assignment)]
            delta[i] = intermediate_data[i] - grid_location

        return grid_assignments, grid_point_probs , grid_locations, delta
    
    def update_grid_assignment(self, n_points, grid_assignments, T, grid_locations, delta):
        new_grid_assignment = np.zeros((n_points, ))
        data_loc = np.zeros((n_points, 2))
        for i in range(n_points):
            grid_assignment = int(grid_assignments[i])
            #new_grid_assignment[i] = np.argmax(T[:,grid_assignment])
            new_grid_assignment[i] = np.random.choice(len(T[:,grid_assignment]), p = T[:, grid_assignment]/np.sum(T[:, grid_assignment]))
            data_loc[i] = grid_locations[int(new_grid_assignment[i])] +  delta[i]

        return new_grid_assignment, data_loc

if __name__ == "__main__":
    # test the UMAP2 class
    # set numpy random seed
    np.random.seed(1)
    target_letter = "U"
    umap2 = UMAP2(target_letter, target_size=12, intermediate_granularity=12)
    reduced = umap2.fit_transform(np.random.rand(800, 30))
    # plot reduced
    plt.scatter(reduced[:,0], reduced[:,1], alpha = 0.2)
    # save the scatter plot
    plt.savefig('umap2.png')

    # clear the figure 
    plt.clf()
    # save the target image 
    plt.imshow(umap2.target_img, cmap = 'gray')
    plt.savefig('target_image.png')
