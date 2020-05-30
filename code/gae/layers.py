from gae.initializations import *
import tensorflow as tf
from sklearn.preprocessing import normalize

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems, dtype=tf.float32):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = tf.cast(keep_prob, dtype=dtype)
    random_tensor += tf.random_uniform(noise_shape, dtype=dtype)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return tf.cast(pre_out, dtype) * tf.cast((1./keep_prob), dtype)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, dtype=tf.float32, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, dtype=dtype, name="weights")
        self.dropout = dropout
        self.adj = adj
        if type(self.adj) == tf.SparseTensor: # convert to dense if necessary
            self.adj = tf.sparse_tensor_to_dense(self.adj, validate_indices=False)
        self.act = act
        self.dtype=dtype

    # Apply Graph Convolution operation:
        # H_1 = activation(A_norm * X * W)
    def _call(self, inputs):
        x = tf.cast(inputs, self.dtype)
        if type(x) == tf.SparseTensor: # convert to dense if necessary
            x = tf.sparse_tensor_to_dense(x, validate_indices=False)
        x = tf.nn.dropout(x, tf.cast(1-self.dropout, self.dtype))
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, dtype=tf.float32, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        self.ir = np.sqrt(6.0 / (input_dim + output_dim))
        self.initial = tf.random_uniform([input_dim, output_dim], minval=-self.ir,
                                maxval=self.ir, dtype=dtype)
        self.vars['weights'] = self.initial
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
        self.dtype=dtype

    # Apply Graph Convolution operation:
        # H_1 = activation(A_norm * X * W)
    def _call(self, inputs):
        x = inputs
        if type(x) == tf.SparseTensor: # convert to dense if necessary
            x = tf.sparse_tensor_to_dense(x, validate_indices=False)
        self.dx = tf.nn.dropout(x, tf.cast(1-self.dropout, self.dtype))
        self.wdx = tf.matmul(tf.cast(self.dx, tf.float32), tf.cast(self.vars['weights'], tf.float32))
        self.awdx = tf.sparse_tensor_dense_matmul(tf.cast(self.adj, tf.float32), tf.cast(self.wdx, tf.float32))
        self.outputs = tf.cast(self.act(self.awdx), self.dtype)
        return self.outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, flatten=True, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.flatten = flatten

    # Reconstruct adjacency matrix from node embeddings:
        # A_pred = activation(Z*Z^T)
        # Simple inner product
    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        if self.flatten == True:
            x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs

class GravityInspiredDecoder(Layer):
    """Gravity-Inspired asymmetric decoder for directed link prediction."""
    def __init__(self, input_dim, normalize=False, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(GravityInspiredDecoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.dropout = dropout
        self.act = act
        self.normalize = normalize

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        # Embedding vector = all dimensions on input except the last
        # Mass parameter = last dimension of input
        if self.normalize:
            inputs_z = tf.math.l2_normalize(inputs[:,0:(self.input_dim - 1)],
                                            axis = 1)
        else:
            inputs_z = inputs[:, 0:(self.input_dim - 1)]
        # Get pairwise node distances in embedding
        dist = pairwise_distance(inputs_z)
        # Get mass parameter
        inputs_mass = inputs[:,(self.input_dim - 1):self.input_dim]
        mass = tf.matmul(tf.ones([tf.shape(inputs_mass)[0],1]),tf.transpose(inputs_mass))
        # Gravity-Inspired decoding
        outputs_1 = mass - tf.scalar_mul(1, tf.log(dist))
        outputs_2 = tf.reshape(outputs_1,[-1])
        outputs = self.act(outputs_2)
        return outputs, outputs_1, outputs_2, mass, inputs_mass, dist

def pairwise_distance(X, epsilon=0.01):
    """ Computes pairwise distances between node pairs
    :param X: n*d embedding matrix
    :param epsilon: add a small value to distances for numerical stability
    :return: n*n matrix of squared euclidean distances
    """
    x1 = tf.reduce_sum(X * X, 1, True)
    x2 = tf.matmul(X, tf.transpose(X))
    # Add epsilon to distances, avoiding 0 or too small distances leading to
    # numerical instability in gravity decoder due to logarithms
    return tf.math.abs(x1 - 2 * x2 + tf.transpose(x1)) + epsilon
