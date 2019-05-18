import tensorflow as tf
import numpy as np
from utils import vectorization

class Model():
    def __init__(self, args):
        def bivariate_gaussian(x1, x2, mu1, mu2, sigma1, sigma2, rho):
            z = tf.square((x1 - mu1) / sigma1) + tf.square((x2 - mu2) / sigma2) \
                - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
            return tf.exp(-z / (2 * (1 - tf.square(rho)))) / \
                   (2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho)))

        def expand(x, dim, N):
            return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], dim)

        if args.action == 'train':
            args.b == 0
        self.args = args

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.T, 3]) # args.T=300 if train else 1, (batch_size, T, 3)
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.T, 3])

        x = tf.split(self.x, args.T, 1) # (T, batch_size, 1, 3)
        x_list = [tf.squeeze(x_i, [1]) for x_i in x] # (T, batch_size, 3)
        def create_lstm_cell(lstm_size):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
            return lstm_cell
        self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([create_lstm_cell(args.rnn_state_size) for _ in range(args.num_layers)])


        self.init_state = self.stacked_cell.zero_state(args.batch_size, tf.float32)

        self.output_list, self.final_state = tf.nn.dynamic_rnn(self.stacked_cell, tf.transpose(x_list, perm=[1, 0, 2]), initial_state=self.init_state)
        NOUT = 1 + args.M * 6  # end_of_stroke, num_of_gaussian * (pi + 2 * (mu + sigma) + rho)
        output_w = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[args.rnn_state_size, NOUT])) # args.rnn_state_size=400
        output_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[NOUT]))

        self.output = tf.nn.xw_plus_b(tf.reshape(tf.concat(self.output_list, 1), [-1, args.rnn_state_size]), 
                                      output_w, output_b) # (batch_size, NOUT=121)
        y1, y2, y_end_of_stroke = tf.unstack(tf.reshape(self.y, [-1, 3]), axis=1)

        self.end_of_stroke = 1 / (1 + tf.exp(self.output[:, 0])) # (?,), 
        pi_hat, self.mu1, self.mu2, sigma1_hat, sigma2_hat, rho_hat = tf.split(self.output[:, 1:], 6, 1)
        pi_exp = tf.exp(pi_hat * (1 + args.b)) # args.b=3
        pi_exp_sum = tf.reduce_sum(pi_exp, 1)
        self.pi = pi_exp / expand(pi_exp_sum, 1, args.M)
        self.sigma1 = tf.exp(sigma1_hat - args.b)
        self.sigma2 = tf.exp(sigma2_hat - args.b)
        self.rho = tf.tanh(rho_hat)
        self.gaussian = self.pi * bivariate_gaussian(
            expand(y1, 1, args.M), expand(y2, 1, args.M),
            self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho
        )
        eps = 1e-20
        self.loss_gaussian = tf.reduce_sum(-tf.log(tf.reduce_sum(self.gaussian, 1) + eps))
        self.loss_bernoulli = tf.reduce_sum(
            -tf.log((self.end_of_stroke + eps) * y_end_of_stroke # e_t * (x_{t+1})_3 + (1 - e_t) * (1 - (x_{t+1})_3)
                    + (1 - self.end_of_stroke + eps) * (1 - y_end_of_stroke))
        )

        self.loss = (self.loss_gaussian + self.loss_bernoulli) / (args.batch_size * args.T)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def sample(self, sess, length, str=None):
        x = np.zeros([1, 1, 3], np.float32)
        x[0, 0, 2] = 1 # The first point state is set to be 1
        strokes = np.zeros([length, 3], dtype=np.float32)
        strokes[0, :] = x[0, 0, :]
        if self.args.mode == 'predict':
            state = sess.run(self.stacked_cell.zero_state(1, tf.float32))
        if self.args.mode == 'synthesis':
            cell1_state = sess.run(self.cell1.zero_state(1, tf.float32))
            cell2_state = sess.run(self.cell2.zero_state(1, tf.float32))
            w = sess.run(tf.zeros([1, self.args.c_dimension]))
            kappa = sess.run(tf.zeros([1, self.args.K, 1]))
            w_list = []
            phi_list = []
            kappa_list = []
        for i in range(length - 1):
            feed_dict = {self.x: x, self.init_state: state}
            end_of_stroke, pi, mu1, mu2, sigma1, sigma2, rho, state = sess.run(
                [self.end_of_stroke, self.pi, self.mu1, self.mu2,
                 self.sigma1, self.sigma2, self.rho, self.final_state],
                feed_dict=feed_dict
            )
            x = np.zeros([1, 1, 3], np.float32)
            r = np.random.rand()
            accu = 0
            for m in range(self.args.M):
                accu += pi[0, m]
                if accu > r:
                    x[0, 0, 0:2] = np.random.multivariate_normal(
                        [mu1[0, m], mu2[0, m]],
                        [[np.square(sigma1[0, m]), rho[0, m] * sigma1[0, m] * sigma2[0, m]],
                         [rho[0, m] * sigma1[0, m] * sigma2[0, m], np.square(sigma2[0, m])]]
                    )
                    break
            e = np.random.rand() # bernouli
            if e < end_of_stroke:
                x[0, 0, 2] = 1
            else:
                x[0, 0, 2] = 0
            strokes[i + 1, :] = x[0, 0, :]
        return strokes
