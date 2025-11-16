import pandas as pd
import numpy as np
from scipy.stats import mode

data = pd.read_csv("data.csv")
true_labels = pd.read_csv("label.csv")
data = data.dropna(axis=1)

# initialize random centroids
def init_centroids(data, k, global_seed=50):
    centroids = []
    for i in range(k):
        seed = global_seed + i
        # iterate over all columns and randomly pick 1 value to intiialize the initial centroid
        centroid = data.apply(lambda x: float(x.sample(random_state=seed)))
        centroids.append(centroid)
    initial_centroids = pd.concat(centroids, axis=1).T # each column in this dataframe is a centroid, with ecah row beign a different feature
    return initial_centroids

centroids_df = init_centroids(data, 10, global_seed=50) # intialize k = 10 random centroids here

# distance measures

def euclidean_distance(p1, p2):
    # p1 = row from data table
    # p2 = row from centroid
    euc_dist = np.sqrt(np.sum((p1-p2)**2))
    return euc_dist
def cosine_distance(p1,p2):
    # defined as 1 - cosine similarity
    numerator = np.dot(p1, p2)
    denominator = np.sqrt(np.sum(p1 ** 2)) * np.sqrt(np.sum(p2 ** 2))
    if denominator == 0:
        return 1
    cosine_similarity = numerator / denominator
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def jaccard_distance(p1,p2):
    numerator = np.sum(np.minimum(p1, p2)) # the interection between p1 and p2
    denominator = np.sum(np.maximum(p1, p2)) # the union between p1 and p2
    if denominator == 0:
        return 0
    jaccard_distance = 1 - (numerator / denominator)
    return jaccard_distance



def distances(points_df, centroids_df, metric):
    distances = []
    for index1, point in points_df.iterrows():
        point_dist = [] # distance of point from each centroid
        for index2, centroid in centroids_df.iterrows():
            if metric == "euclidean":
                dist = euclidean_distance(point, centroid)
            elif metric == "cosine":
                dist = cosine_distance(point, centroid)
            elif metric == "jaccard":
                dist = jaccard_distance(point, centroid)
            point_dist.append(dist)
        distances.append(point_dist)
    distances_df = pd.DataFrame(distances, index = points_df.index, columns = [f"C{j}" for j in range (10)])
    return distances_df

# kmeans algorithm

def kmeans(points_df, centroids_df, metric):
    centroids = centroids_df.copy().reset_index(drop=True)
    sse = []
    max_iter = 100
    for iter in range(max_iter):
        # compute distances
        distance_df = distances(points_df, centroids, metric)

        # use the distance df to label points
        point_labels = distance_df.idxmin(axis="columns") # in each row, get column label with lowest value
        '''
        Example output
        0 c0
        1 c3
        2 c5
        '''
        # compute the new centroids
        new_centroids = []
        for j in range(10):
            cluster_points = points_df[point_labels == f"C{j}"] # get all points associated with centroid j
            if not cluster_points.empty:
                n_cent = cluster_points.mean().to_numpy()
                new_centroids.append(n_cent)
            else: # if empty
                new_centroids.append(points_df.sample(1).to_numpy().flatten())
        new_centroids = pd.DataFrame(new_centroids, columns=points_df.columns)


        # sse calculation accounting for metric
        sse_val = 0
        for index, point in points_df.iterrows():
            cluster_id = int(point_labels[index][1:]) # convert the string label into an int representing the cluster
            centroid = new_centroids.loc[cluster_id].values

            j_dist = jaccard_distance(point.values, centroid)
            sse_val += j_dist ** 2

        sse.append(sse_val)

        # Stopping condition checks
        #############################################################################################################
        # 1. check for convergence when centroids dont change
        tol = 0
        change = np.sqrt(np.sum((centroids.to_numpy() - new_centroids.to_numpy()) ** 2))
        if change == tol:
            print(f"Converged after {iter} iteration")
            break

        # 2. check for convergence when sse values dont change
        # if len(sse)>3:
        #     if sse[-1] > sse[-2]:
        #         print(f"sse val in iteration {iter} increased over last iteration")
        #         sse.pop()
        #         break
        #     elif sse[-1] == sse[-2] == sse[-3]:
        #         same_sse+=1
        #         print(f"sse val in iteration {iter} same as last 3 iterations")
        #         if same_sse >= 5:
        #             break

        # 3. To run for max_iterations, comment both 1 and 2 above
        #################################################################################################################      
        centroids = new_centroids.copy()

    return point_labels, centroids, distance_df, sse

pred_labels_jac, final_centroids_jac, dist_matrix_jac, sse_history_jac = kmeans(data, centroids_df, "jaccard")
print("Using Jaccard Distance:\n\n\n")

print("\nCluster Assignments:")
print(pred_labels_jac)
print("\nFinal Centroids:")
print(final_centroids_jac)
# print("\nSSE per iteration:")
# print(sse_history)
print(f"\nFinal SSE: {sse_history_jac[-1]}")

# kmeans accuracy for euclidean

def kmeans_acc(true_labels, pred_labels):
    # flattent the dataframe to convert to series
    if isinstance(true_labels, pd.DataFrame):
        true_labels = true_labels.values.flatten()
    if isinstance(pred_labels, pd.DataFrame):
        pred_labels = pred_labels.vlaues.flatten()
    true_labels = pd.Series(true_labels).reset_index(drop=True)
    pred_labels = pd.Series(pred_labels).reset_index(drop=True)

    # convert labels to int for each point
    pred_labels_int = pred_labels.apply(lambda x: int(x[1:]))

    # create a dict that maps cluster to most frequent class for true labels
    label_dict = {}
    for cluster_id in range(10): # for the clusters 0 to 9
        mode_label = mode(true_labels[pred_labels_int == cluster_id], keepdims=True).mode[0]
        label_dict[cluster_id] = mode_label
        print(f"Cluster {cluster_id} has label {mode_label}")

    map_pred = pred_labels_int.map(label_dict)

    accuracy = (map_pred == true_labels).mean()
    return accuracy, label_dict

# accuracy for jaccard
acc_jac, map_jac = kmeans_acc(true_labels, pred_labels_jac)
print(f"Jaccard Accuracy: {acc_jac}\n\n")
print("True Mapping of cluster - label:", map_jac)