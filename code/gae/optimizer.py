import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Graph AE: use Weighted-cross-entropy loss
class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm, learning_rate=0.001):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


# Graph VAE: use weighted-cross-entropy loss + KL Divergence
class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, learning_rate=0.001, dtype=tf.float32):
        self.preds_sub = preds
        self.labels_sub = labels

        self.i = model.inputs
        self.adj = model.adj
        # self.ir = model.ir
        self.initial = model.initial
        self.fw = model.fw
        self.dx = model.dx
        self.wdx = model.wdx
        self.awdx = model.awdx
        self.h1 = model.hidden1
        self.zm = model.z_mean
        self.zls = model.z_log_std
        self.z = model.z
        self.r = model.reconstructions
        self.m = model.m
        self.im = model.im
        self.d = model.d
        self.o1 = model.o1
        self.o2 = model.o2
        self.num_nodes = num_nodes

        print('Creating GAE optimizer...')
        print('Labels shape: ', self.labels_sub.shape)
        print('Preds shape: ', self.preds_sub.shape)

        self.a = tf.nn.weighted_cross_entropy_with_logits(logits=self.preds_sub, targets=self.labels_sub, pos_weight=pos_weight)
        self.b = tf.reduce_mean(self.a)

        self.cost = norm * self.b
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / self.num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        print('CE+KL loss shape: ', self.cost.shape)

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        if dtype == tf.float32:
            self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(self.preds_sub), 0.5), tf.int32),
                                            tf.cast(self.labels_sub, tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        elif dtype == tf.float16:
            # self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(self.preds_sub), 0.5), tf.int16),
            #                                 tf.cast(self.labels_sub, tf.int16))
            self.accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(
                        tf.cast(
                            tf.greater_equal(tf.sigmoid(self.preds_sub), 0.5), tf.int16),
                            tf.cast(self.labels_sub, 
                        tf.int16)), 
                    tf.float16))
