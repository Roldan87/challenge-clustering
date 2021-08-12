import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import KMeans, AgglomerativeClustering, FeatureAgglomeration, MeanShift, SpectralClustering, estimate_bandwidth
from itertools import combinations


# Read DF
df_bear = pd.read_csv('bearing_final_data.csv', index_col=0)

# Function to iterate through all possible combinations between 2 features
def iter_feature(arr, r):
    return list(combinations(arr, r))

# Get Column names in dataset as list
arr = df_bear.columns.to_list()

#Function for Silhouette scores and Plots
def scores_and_plots(cluster_model, model_name, df, feat_one, feat_two):
    # Create a subplot with 1 row and 2 columns 
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    # Clustering Model fit
    clusterer = cluster_model
    cluster_labels = clusterer.fit_predict(df)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(df, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(df_vib_amp[feat_one], df_vib_amp[feat_two], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    # Labeling the clusters
    centers = clusterer.cluster_centers_
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle((f"Silhouette analysis for {model_name} clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')


# KMEANS CLUSTERING MODEL
# Cluster all combinations of 2 features & Silhouette Score & Plots
r = 2
feature_comb = iter_feature(arr,r)
for index, tup in enumerate(feature_comb):
    element_one = tup[0]
    element_two = tup[1]
    df_vib_amp = df_bear[element_one].to_frame().join(df_bear[element_two].to_frame())
    df_vib_amp = df_vib_amp.reset_index()
    print(f"Features: {element_one} and {element_two}")

    model_name = 'KMeans'
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10)
    df = df_vib_amp
    range_n_clusters = [2, 3, 4, 5, 6, 7]
    for n_clusters in range_n_clusters:
        scores_and_plots(cluster_model, model_name, df, element_one, element_two)


# Cluster all combinations of 3 to 6 features & Silhouette Score & Plots (2 first features only)
r = 3
feature_comb = iter_feature(arr,r)
for index, tup in enumerate(feature_comb):
    element_one = tup[0]
    element_two = tup[1]
    element_three = tup[2]
    df_vib_amp = df_bear[element_one].to_frame().join(df_bear[element_two].to_frame()).join(df_bear[element_three].to_frame())
    df_vib_amp = df_vib_amp.reset_index()
    print(f"Features: {element_one} and {element_two} and {element_three}")

    model_name = 'KMeans'
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10)
    df = df_vib_amp
    range_n_clusters = [2, 3, 4, 5, 6, 7]
    for n_clusters in range_n_clusters:
        scores_and_plots(cluster_model, model_name, df, element_one, element_two)

r = 4
feature_comb = iter_feature(arr,r)
for index, tup in enumerate(feature_comb):
    element_one = tup[0]
    element_two = tup[1]
    element_three = tup[2]
    element_four = tup[3]
    df_vib_amp = df_bear[element_one].to_frame().join(df_bear[element_two].to_frame()).join(df_bear[element_three].to_frame()).join(df_bear[element_four].to_frame())
    df_vib_amp = df_vib_amp.reset_index()
    print(f"Features: {element_one} and {element_two} and {element_three} and {element_four}")

    model_name = 'KMeans'
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10)
    df = df_vib_amp
    range_n_clusters = [2, 3, 4, 5, 6, 7]
    for n_clusters in range_n_clusters:
        scores_and_plots(cluster_model, model_name, df, element_one, element_two)

r = 5
feature_comb = iter_feature(arr,r)
for index, tup in enumerate(feature_comb):
    element_one = tup[0]
    element_two = tup[1]
    element_three = tup[2]
    element_four = tup[3]
    element_five = tup[4]
    df_vib_amp = df_bear[element_one].to_frame().join(df_bear[element_two].to_frame()).join(df_bear[element_three].to_frame()).join(df_bear[element_four].to_frame()).join(df_bear[element_five].to_frame())
    df_vib_amp = df_vib_amp.reset_index()
    print(f"Features: {element_one} and {element_two} and {element_three} and {element_four} and {element_five}")

    model_name = 'KMeans'
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10)
    df = df_vib_amp
    range_n_clusters = [2, 3, 4, 5, 6, 7]
    for n_clusters in range_n_clusters:
        scores_and_plots(cluster_model, model_name, df, element_one, element_two)

r = 6
feature_comb = iter_feature(arr,r)
for index, tup in enumerate(feature_comb):
    element_one = tup[0]
    element_two = tup[1]
    element_three = tup[2]
    element_four = tup[3]
    element_five = tup[4]
    element_six = tup[5]
    df_vib_amp = df_bear[element_one].to_frame().join(df_bear[element_two].to_frame()).join(df_bear[element_three].to_frame()).join(df_bear[element_four].to_frame()).join(df_bear[element_five].to_frame()).join(df_bear[element_six].to_frame())
    df_vib_amp = df_vib_amp.reset_index()
    print(f"Features: {element_one} and {element_two} and {element_three} and {element_four} and {element_five} and {element_six}")

    model_name = 'KMeans'
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10)
    df = df_vib_amp
    range_n_clusters = [2, 3, 4, 5, 6, 7]
    for n_clusters in range_n_clusters:
        scores_and_plots(cluster_model, model_name, df, element_one, element_two)


# MODEL COMPARISON
# Reset df with 2 best features for model comparison
df_compare = df_bear['a2_x_mean'].to_frame().join(df_bear['a2_x_amp_max'].to_frame())
df_compare = df_compare.reset_index()
feature_one = 'a2_x_mean'
feature_two = 'a2_x_amp_max'
print(f"Features: a2_x_mean and a2_x_amp_max")


# AGGLOMERATIVE CLUSTERING
df = df_compare
model_name = 'AgglomerativeClustering'
cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
range_n_clusters = [2, 3, 4, 5, 6, 7]
for n_clusters in range_n_clusters:
    scores_and_plots(cluster_model, model_name, df, feature_one, feature_two)


# MEANSHIFT CLUSTERING
df = df_compare
model_name = 'MeanShift Clustering'
bandwidth = estimate_bandwidth(df, quantile=0.2, n_samples=600)
cluster_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
scores_and_plots(cluster_model, model_name, df, feature_one, feature_two)


# SPACTRAL CLUSTERING
df = df_compare
model_name = 'Spactral Clustering'
cluster_model = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=42)
range_n_clusters = [2, 3, 4, 5, 6, 7]
for n_clusters in range_n_clusters:
    scores_and_plots(cluster_model, model_name, df, feature_one, feature_two)