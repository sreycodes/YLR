from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt 
import logging

import time
import os
from math import isnan, log, exp
import pickle

import tensorflow as tf
tf.set_random_seed(0)
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Calculate ROC AUC
def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)
        # print(emb)

    def sigmoid(x):
        #Numerically stable sigmoid function
        if x >= 0:
            z = exp(-x)
            res = 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = exp(x)
            res = z / (1 + z)
        return res

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]])) # predicted score for given edge
        pos.append(adj_orig[e[0], e[1]]) # actual value (1)

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]])) # predicted score for given edge
        neg.append(adj_orig[e[0], e[1]]) # actual value (0)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_dataset(datafile_path, num_related, num_rows=None):
    related_columns = ['related' + str(i) for i in range(1, num_related + 1)]
    df = pd.read_csv(datafile_path, delim_whitespace=True, header=None, nrows=num_rows,
        names=['videoid', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'rating', 'comments'] + related_columns)
    logging.debug('Dataset read')
    df = df.fillna(df.mean())
    df = df.fillna(df.mode().iloc[0])
    category_columns = ['category_' + str(x) for x in df.category.unique()]
    df['cat_orig'] = df['category']
    df = pd.get_dummies(df, columns=['category'], prefix='category')
    df = df.rename(columns={'cat_orig': 'category'})
    df['row_number'] = np.arange(len(df))
    return df, category_columns

def get_graph_and_features(df, num_related, cat_cols, adj_file_path=None, features_file_path=None, mode=None): #Here num_related is how many to consider out of all given in dataset
    if adj_file_path and features_file_path and mode is 'r':
        return sp.load_npz(adj_file_path), sp.load_npz(features_file_path)
    else:
        videoids = df.videoid.unique()
        d = df.set_index('videoid').to_dict()
        videoid_with_row = d['row_number']
        num_features = len(['age', 'length', 'views', 'rate', 'rating', 'comments'] + cat_cols)
        # videoid_with_views = d['views']
        # videoid_with_rating = d['rating']

        adj = sp.lil_matrix((len(videoids), len(videoids)))
        features = sp.lil_matrix((len(videoids), num_features))
        cat_count = {}
        cat_out_edges = {}
        for index, row in df.iterrows():
            f = row[['age', 'length', 'views', 'rate', 'rating', 'comments'] + cat_cols].astype('float64').to_list()
            features[index, :] = f
            if row.category in cat_count.keys():
                cat_count[row.category] += 1
            else:
                cat_count[row.category] = 1
            if row.category not in cat_out_edges:
                cat_out_edges[row.category] = 0
            for i in range(1, 1 + num_related):
                node = row['related' + str(i)]
                if node in videoids:
                    row_index = videoid_with_row[node]
                    adj[index, row_index] = 1 #unweighted
                    adj[row_index, index] = 1 #undirected
                    node_row = df.iloc[row_index]
                    if row.category != node_row.category:
                        cat_out_edges[row.category] += 1
            adj[index, index] = 1 # Set self loop

        # print(cat_count)
        # print(cat_out_edges)

        logging.debug('Rows processed')
        features = normalize(features, axis=0)
        logging.debug('Normalized features')

        if adj_file_path and features_file_path and mode is 'w':
            sp.save_npz(adj_file_path, adj.tocsr())
            sp.save_npz(features_file_path, features.tocsr())
            print("Saved stuff")

        return adj, features

def save_visualization(g, file_name, title):
    plt.figure(figsize=(18,18))
    degrees = dict(nx.degree(g))
    
    # Draw networkx graph -- scale node size by log(degree+1)
    nx.draw_spring(g, with_labels=False, 
                   linewidths=2.0,
                   nodelist=degrees.keys(),
                   node_size=[log(degree_val+1) * 100 for degree_val in degrees.values()], \
                   node_color='r')
    
    # Create black border around node shapes
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    
#     plt.title(title)
    plt.savefig(file_name)
    plt.clf()

def get_network_statistics(g):
    num_connected_components = nx.number_connected_components(g)
    num_nodes = nx.number_of_nodes(g)
    num_edges = nx.number_of_edges(g)
    density = nx.density(g)
    avg_clustering_coef = nx.average_clustering(g)
    avg_degree = sum(dict(g.degree()).values()) / float(num_nodes)
    transitivity = nx.transitivity(g)
    
    if num_connected_components == 1:
        diameter = nx.diameter(g)
    else:
        diameter = None # infinite path length between connected components
    
    network_statistics = {
        'num_connected_components':num_connected_components,
        'num_nodes':num_nodes,
        'num_edges':num_edges,
        'density':density,
        'diameter':diameter,
        'avg_clustering_coef':avg_clustering_coef,
        'avg_degree':avg_degree,
        'transitivity':transitivity
    }
    
    return network_statistics

def save_network_statistics(g, file_name):
    network_statistics = get_network_statistics(g)
    print(network_statistics)
    with open(file_name, 'wb') as f:
        pickle.dump(network_statistics, f)

logging.basicConfig(filename='graph_vae.log',level=logging.DEBUG, filemode='w', format='%(asctime)s %(levelname)s:%(message)s')
datafile_path = '../data/youtube-dataset.txt'
num_related = 20
num_rows = None
# df, cat_cols = get_dataset(datafile_path, num_related, num_rows)
# print(df.head())
logging.debug('Dataset obtained')
adj, features = get_graph_and_features(None, num_related, None, 
    adj_file_path='../data/yd-rel-20-adj.npz', features_file_path='../data/yd-rel-20-feat.npz', mode='r')
logging.debug('Generated adjacency_matrix and features')

node_rate = features.getcol(2).toarray().flatten() #Choosing views
k = 1000
top_nodes = node_rate.argsort()[-k:].tolist()
all_nodes = set()
for node in top_nodes:
    all_nodes.add(node)
    # neighbours = adj.getrow(node).nonzero()[1]
    # all_nodes.update(neighbours)
all_nodes = np.array(list(all_nodes))
adj = adj[all_nodes[:, None], all_nodes]
features = features[all_nodes, :]
logging.debug('Reduced dataset')
# g = nx.from_scipy_sparse_matrix(adj)
# logging.debug('Made graph')
# save_visualization(g, '../data/viz-rel-1.png', 'Viz')
# logging.debug('Saved viz')
# save_network_statistics(g, '../data/stats-rel-1.info')
# logging.debug('Saved network statistics')
print(features.shape)
print(adj.shape)

# _ = input("Press any key to continue\n")

features_tuple = sparse_to_tuple(features)
features_shape = features_tuple[2]

# Get graph attributes (to feed into model)
num_nodes = adj.shape[0] # number of nodes in adjacency matrix
num_features = features_shape[1] # number of features (columsn of features matrix)
features_nonzero = features_tuple[1].shape[0] # number of non-zero entries in features matrix (or length of values list)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

degrees = np.squeeze(np.asarray(adj.sum(axis=1).flatten()))
degrees = degrees ** -0.5
deg_matrix = sp.diags([degrees], [0])

logging.debug('Some preprocessing done')

np.random.seed(0) # IMPORTANT: guarantees consistent train/test splits
adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = mask_test_edges(adj, test_frac=.3, val_frac=.1, verbose=True)

logging.debug('Splitting done')

# print(adj_train.shape)
# Normalize adjacency matrix
adj_norm = normalize(adj_train, axis=1)
adj_norm = deg_matrix.tocsr() * adj_norm.tocsr() * deg_matrix.tocsr()
adj_norm = sparse_to_tuple(adj_norm)
logging.debug('Preprocessed graph')

# Add in diagonals
adj_label_mat = adj_orig + sp.eye(adj_orig.shape[0])
adj_label = sparse_to_tuple(adj_label_mat)

# Inspect train/test split
print("Total nodes:", adj.shape[0])
print("Total edges:", int(adj.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
print("Training edges (positive):", len(train_edges))
print("Training edges (negative):", len(train_edges_false))
print("Validation edges (positive):", len(val_edges))
print("Validation edges (negative):", len(val_edges_false))
print("Test edges (positive):", len(test_edges))
print("Test edges (negative):", len(test_edges_false))

# Define hyperparameters
LEARNING_RATE = 0.005
EPOCHS = 300
HIDDEN1_DIM = 32
HIDDEN2_DIM = 16
DROPOUT = 0.1

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
    # 'features_nonzero': tf.placeholder_with_default(features_nonzero, shape=()),
    # 'num_nodes': tf.placeholder_with_default(num_nodes, shape=())
}

# How much to weigh positive examples (true edges) in cost print_function
  # Want to weigh less-frequent classes higher, so as to prevent model output bias
  # pos_weight = (num. negative samples / (num. positive samples)
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

# normalize (scale) average weighted cost
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
 
# Create VAE model
model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero,
                   HIDDEN1_DIM, HIDDEN2_DIM)

opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           learning_rate=LEARNING_RATE)

cost_val = []
acc_val = []
val_roc_score = []

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

logging.debug('Initialized variables')

# Train model
for epoch in range(EPOCHS):

    start = time.time()

    feed_dict = construct_feed_dict(adj_norm, adj_label, features_tuple, placeholders)
    feed_dict.update({placeholders['dropout']: DROPOUT})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)

    # Print results for this epoch
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - start))

logging.debug("Optimization Finished!")

# Print final results
roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
