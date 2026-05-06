import numpy as np
from sklearn.metrics import silhouette_score

class F3KM:

    def __init__(self, n_clusters,type_balance, type_init, type_update_centroids, max_iter=200, random_state=None, lambda_=0.5):
        self.n_clusters = n_clusters
        self.type_of_balance = type_balance #'norm'/ 'unorm
        self.type_init = type_init #'random_centroids' / 'random_labels' / 'random_c_kmeans'/ 'random_c_kmeans_pp 
        self.type_of_update_centroids = type_update_centroids #'standard' / 'weighted' / 'shifted' / 'simple_weighted' 
        self.max_iter = max_iter
        self.random_state = random_state
        self.lambda_ = lambda_
        self.centroids = None
        self.labels_ = None
        self.cluster_attributes = None
        self.attributes = None 
        self.stop_flag = None
        self.iter = 0
        self.Red = 0
        self.Blue = 0


    def find_clusters_r_b(self, attributes, labels):
        self.cluster_attributes = {}
        
        counts_0 = np.bincount(labels[attributes == 0], minlength=self.n_clusters)
        counts_1 = np.bincount(labels[attributes == 1], minlength=self.n_clusters)
        
        for i in range(self.n_clusters):
            c0 = counts_0[i]
            c1 = counts_1[i]
            self.cluster_attributes[i] = {
                '0': c0,
                '1': c1,
                'majority': 0 if c0 > c1 else 1
            }

    def find_data_Red_Blue(self, attributes):
        self.Red = np.sum(attributes == 0)
        self.Blue = np.sum(attributes == 1)
        #print(f"Red: {self.Red}, Blue: {self.Blue}")
        return self.Red, self.Blue
    
    # ties random selection
    def random_argmax(self, array):
        max_vals = np.max(array, axis=1, keepdims=True)
        is_max = (array == max_vals)
        choices = [np.random.choice(np.flatnonzero(row)) for row in is_max]
        return np.array(choices)
    
    def random_init_centroids(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        return X[random_idx[:self.n_clusters]]
    
    def update_centroids(self, X):
        epsilon = 1e-8
        new_centroids = []
        wi = None
        for j in range(self.n_clusters):
            mask = (self.labels_ == j)
            cluster_points = X[mask]
            cluster_attrs = self.attributes[mask]

            if len(cluster_points) == 0:
                new_centroids.append(self.centroids[j])
                continue

            if self.cluster_attributes is not None and j in self.cluster_attributes:
                count_0 = self.cluster_attributes[j]['0']
                count_1 = self.cluster_attributes[j]['1']
            else:
                count_0 = np.sum(cluster_attrs == 0)
                count_1 = np.sum(cluster_attrs == 1)
            total = len(cluster_points)

            #1 standard centroid update
            if self.type_of_update_centroids == "standard":
                centroid = np.mean(cluster_points, axis=0)
                new_centroids.append(centroid)
                continue
            
            # 2 Simple-weighted Centroid Update
            # if self.type_of_update_centroids == "simple_weighted":
            #     w = np.where(cluster_attrs == 0,
            #                  1.0 / (count_0 + epsilon),
            #                  1.0 / (count_1 + epsilon))
            #     centroid = np.average(cluster_points, axis=0, weights=w)
            #     new_centroids.append(centroid)
            #     continue

            # 3 Shifted Centroid Update closer to minority points
            # elif self.type_of_update_centroids == "shifted":
            #     mean_cluster = np.mean(cluster_points, axis=0)
            #     if count_0 == 0 or count_1 == 0 or count_0 == count_1:
            #         new_centroids.append(mean_cluster)
            #         continue
            #     minority_attr = 0 if count_0 < count_1 else 1
            #     minority_points = cluster_points[cluster_attrs == minority_attr]
            #     mean_minority = np.mean(minority_points, axis=0)
            #     shifted_centroid = mean_cluster + self.lambda_ * mean_minority
            #     new_centroids.append(shifted_centroid)
            #     # print(f"mean_cluster: {mean_cluster}, mean_minority: {mean_minority}, shifted_centroid: {shifted_centroid}")
            #     # print(f"count_0: {count_0}, count_1: {count_1}, minority_attr: {minority_attr}")
            #     continue

            # 4 Weighted Centroid Update based on inverse of imbalance
            elif self.type_of_update_centroids == "weighted":
                if(wi is None):
                    wi = self.compute_weights(X)
                cluster_wi = wi[mask, j]
                abs_w = np.abs(cluster_wi)
                weights = np.where(abs_w == 0, epsilon, abs_w)
                centroid = np.average(cluster_points, axis=0, weights=weights)
                new_centroids.append(centroid)
                continue

            else:
                raise ValueError(f"Unknown mode: {self.type_of_update_centroids}")

        return np.array(new_centroids)

    def initialize_points(self, X, attributes):
        if self.type_init == 'random_centroids':
            print("random centroids init")
            self.attributes = attributes.copy()
            self.centroids = self.random_init_centroids(X)#randomly initialize centroids
            self.labels_ = -2 *np.ones(X.shape[0], dtype=int)
            for i, centroid in enumerate(self.centroids):
                mask = np.all(X == centroid, axis=1)
                self.labels_[mask] = i
            self.find_clusters_r_b(self.attributes, self.labels_)

            return 
        
        elif self.type_init == 'random_labels':
            print("random labels init")
            self.attributes = attributes.copy()
            np.random.seed(self.random_state)
            self.labels_ = np.random.randint(0, self.n_clusters, size=X.shape[0])
            self.find_clusters_r_b(self.attributes, self.labels_)
            self.centroids = self.update_centroids(X)
            self.find_clusters_r_b(self.attributes, self.labels_)
            return
        
        elif self.type_init == 'random_c_kmeans':
            self.attributes = attributes.copy()
            self.centroids = self.random_init_centroids(X)
            l = self.lambda_
            self.lambda_ = 0
            new_distances = self.compute_forces(X, self.centroids)
            self.labels_ = np.argmax(new_distances, axis=1)
            self.labels_ = self.random_argmax(new_distances)
            self.find_clusters_r_b(self.attributes, self.labels_)
        
            self.centroids = self.update_centroids(X)  
            self.lambda_ = l 
            return 

        elif self.type_init == 'random_c_kmeans_pp':
            self.attributes = attributes.copy()

            self.centroids = self.random_init_centroids(X)
            l = self.lambda_
            self.lambda_ = 0
            new_distances = self.compute_forces(X, self.centroids)
            self.labels_ = np.argmax(new_distances, axis=1)
            self.labels_ = self.random_argmax(new_distances)
            self.find_clusters_r_b(self.attributes, self.labels_)
            self.centroids = self.update_centroids(X)  
            self.lambda_ = l 
            return
        else:
            raise ValueError(f"Unknown initialization method: {self.type_init}")

    def compute_weights(self, X):
        cluster_counts = np.array([[self.cluster_attributes[i]['0'], self.cluster_attributes[i]['1']]
                                    for i in range(self.n_clusters)])  
        adj_0 = np.broadcast_to(cluster_counts[:,0], (X.shape[0], self.n_clusters))
        adj_1 = np.broadcast_to(cluster_counts[:,1], (X.shape[0], self.n_clusters))
        if self.type_of_balance == 'unorm':
            ratios = np.where((adj_0 == 0) | (adj_1 == 0), 0,
                        np.minimum(adj_0 / np.maximum(adj_1, 1e-10),
                                    adj_1 / np.maximum(adj_0, 1e-10)))
            imbalance = 1.0 - ratios
            majority_adj = np.where(adj_0 > adj_1, 0, np.where(adj_1 > adj_0, 1, -2))

        elif self.type_of_balance == 'norm':
            ratios = np.where((adj_0 == 0) | (adj_1 == 0), 0,
                        np.minimum(
                        (adj_0 / np.maximum(adj_1, 1e-10)) / (self.Red / np.maximum(self.Blue, 1e-10)),
                        (adj_1 / np.maximum(adj_0, 1e-10)) / (self.Blue / np.maximum(self.Red, 1e-10)))
                        )
            imbalance = 1.0 - ratios
            majority_adj = np.where((adj_0 / np.maximum(adj_1, 1e-10)) / (self.Red / np.maximum(self.Blue, 1e-10)) > 1, 0, np.where((adj_1 / np.maximum(adj_0, 1e-10)) / (self.Blue / np.maximum(self.Red, 1e-10))>1, 1, -2))

        direction = np.where(self.attributes[:, None] == majority_adj, -1.0, 1.0)
        direction[majority_adj == -2] = 0
        # print(f'imbalance: {imbalance}')
        # print(f'majority_adj: {majority_adj}')
        # print(f"direction: {direction}")
        # print(f"lambda: {self.lambda_}")
        weights =  ( 1 + self.lambda_ * imbalance * direction)
    
        return weights

    def compute_forces(self, X, centroids):
        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        norms = np.sum(diffs**2, axis=2)
        norms = np.maximum(norms, 1e-10)
        distances = 1.0 / norms

        if self.lambda_ == 0:
            return distances

        cluster_counts = np.array([
            [self.cluster_attributes[i]['0'], self.cluster_attributes[i]['1']]
            for i in range(self.n_clusters)
        ]) 
        adj_0 = np.broadcast_to(cluster_counts[:,0], (X.shape[0], self.n_clusters))
        adj_1 = np.broadcast_to(cluster_counts[:,1], (X.shape[0], self.n_clusters))

        if self.type_of_balance == 'unorm':
            ratios = np.where((adj_0 == 0) | (adj_1 == 0), 0,
                        np.minimum(adj_0 / np.maximum(adj_1, 1e-10),
                                    adj_1 / np.maximum(adj_0, 1e-10)))

            imbalance = 1.0 - ratios
            majority_adj = np.where(adj_0 > adj_1, 0, np.where(adj_1 > adj_0, 1, -2))
        elif self.type_of_balance == 'norm':
            ratios = np.where((adj_0 == 0) | (adj_1 == 0), 0,
                        np.minimum(
                        (adj_0 / np.maximum(adj_1, 1e-10)) / (self.Red / np.maximum(self.Blue, 1e-10)),
                        (adj_1 / np.maximum(adj_0, 1e-10)) / (self.Blue / np.maximum(self.Red, 1e-10)))
                        )

            imbalance = 1.0 - ratios
            majority_adj = np.where((adj_0 / np.maximum(adj_1, 1e-10)) / (self.Red / np.maximum(self.Blue, 1e-10)) > 1, 0, np.where((adj_1 / np.maximum(adj_0, 1e-10)) / (self.Blue / np.maximum(self.Red, 1e-10))>1, 1, -2))

        direction = np.where(self.attributes[:, None] == majority_adj, -1.0, 1.0)
        direction[majority_adj == -2] = 0
        new_metric =  ( 1 + self.lambda_ * imbalance * direction) * distances 
        return new_metric

    def fit(self, X, attributes):
        
        print("All in one fit")
        self.find_data_Red_Blue(attributes)
        print(f"Red: {self.Red}, Blue: {self.Blue}")

        self.initialize_points(X, attributes)
        iter = 0
        prev_centroids = None

        for _ in range(self.max_iter):
            iter +=1 
            new_distances = self.compute_forces(X, self.centroids)
            self.labels_ = np.argmax(new_distances, axis=1)
            self.find_clusters_r_b(self.attributes, self.labels_)
            new_centroids = self.update_centroids(X)

            if np.allclose(new_centroids, self.centroids, atol=1e-6):
                self.stop_flag = "convergence"
                self.iter = iter
                print("convergence")
                break
            if prev_centroids is not None and np.allclose(new_centroids, prev_centroids, atol=1e-6):
                self.stop_flag = "oscillation"
                self.iter = iter
                print(f"Oscillation detected at iter {iter}")
                break

            prev_centroids = self.centroids
            self.centroids = new_centroids 

            if self.max_iter == iter:
                self.stop_flag = "max_iter"
                self.iter = iter
        return self 

    def calculate_ratio(self, adj_0, adj_1):
        if adj_0 == 0 or adj_1 == 0:
            return 0
        else:
            return min(adj_0 / adj_1, adj_1 / adj_0)

    def fairness_metrics(self):
        fairness_values ={'overall':0,
                          'Balance':0,
                          'N_Balance':0,
                          'FR':0, 
                          'FR_norm':0}
        all_Balance_ratios = []
        all_N_Balance_ratios = []
        self.find_data_Red_Blue(self.attributes)
        for i in range(self.n_clusters):
            c0 = self.cluster_attributes[i]['0'] 
            c1 = self.cluster_attributes[i]['1']
            all_Balance_ratios.append(self.calculate_ratio(c0, c1))
            cluster_ratio = min((c0/c1)/(self.Red/self.Blue), (c1/c0)/(self.Blue/self.Red)) if c0 > 0 and c1 > 0 and self.Red > 0 and self.Blue > 0 else 0
            all_N_Balance_ratios.append(cluster_ratio)

        fairness_values['overall'] = self.calculate_ratio(self.Red, self.Blue)
        fairness_values['Balance'] = np.mean(all_Balance_ratios) if all_Balance_ratios else 0
        fairness_values['N_Balance'] = np.mean(all_N_Balance_ratios) if all_N_Balance_ratios else 0
        fairness_values['FR'] = np.min(all_Balance_ratios) if all_Balance_ratios else 0
        fairness_values['FR_norm'] = np.min(all_N_Balance_ratios) if all_N_Balance_ratios else 0     
        return fairness_values
    
    def quality_metrics(self, X):
        quality_values = {'Silhouette': 0,'SSE': 0}
        quality_values['Silhouette'] = silhouette_score(X, self.labels_) if len(set(self.labels_)) > 1 else 0
        quality_values['SSE'] = np.sum((X - self.centroids[self.labels_]) ** 2)
        return quality_values