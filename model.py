#import tensorflow as tf
import torch
import numpy as np
from utils import vectorization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()


        if args.action == 'train':
            args.b == 0
        self.args = args

        self.NOUT = 1 + self.args.M * 6  # end_of_stroke, num_of_gaussian * (pi + 2 * (mu + sigma) + rho)
        self.fc_output = torch.nn.Linear(args.rnn_state_size, self.NOUT)


#        #self.cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_state_size) # args.rnn_state_size=400
#        #self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * args.num_layers) # args.num_layers=2
#        def create_lstm_cell(lstm_size):
#            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
#            return lstm_cell
#        self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([create_lstm_cell(args.rnn_state_size) for _ in range(args.num_layers)])
        self.stacked_cell = torch.nn.LSTM(input_size=3, hidden_size=args.rnn_state_size, num_layers=2, batch_first=True)

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=args.learning_rate)

        
    def fit(self, x, y):
        '''
            x: (batch_size, args.T, 3) # args.T=300 if train else 1, (batch_size, T, 3)
            y: (batch_size, args.T, 3)
        '''
        def bivariate_gaussian(x1, x2, mu1, mu2, sigma1, sigma2, rho):
            z = torch.pow((x1 - mu1) / sigma1, 2) + torch.pow((x2 - mu2) / sigma2, 2) \
                - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
            return torch.exp(-z / (2 * (1 - torch.pow(rho, 2)))) / \
                   (2 * np.pi * sigma1 * sigma2 * torch.sqrt(1 - torch.pow(rho, 2)))
        def expand(x, dim, N):
            #return tf.concat(dim, [tf.expand_dims(x, dim) for _ in range(N)])
            return torch.cat([x.unsqueeze(dim) for _ in range(N)], dim)
        self.x = torch.Tensor(x).to(device)
        self.y = torch.Tensor(y).to(device)

        #x = torch.split(self.x, self.args.T, 1) # (T, batch_size, 1, 3)
        #x_list = [tf.squeeze(x_i, [1]) for x_i in x] # (T, batch_size, 3)
        #x_list = torch.stack([torch.squeeze(x_i, dim=1) for x_i in x]) # (T, batch_size, 3)

#        self.init_state = self.stacked_cell.zero_state(args.batch_size, tf.float32)
#        #self.output_list, self.final_state = tf.nn.rnn(self.stacked_cell, x_list, self.init_state)
#        self.output_list, self.final_state = tf.nn.dynamic_rnn(self.stacked_cell, tf.transpose(x_list, perm=[1, 0, 2]), initial_state=self.init_state)

        #self.output_list, self.final_state = self.stacked_cell(torch.transpose(x_list, 0, 1))
        self.output_list, self.final_state = self.stacked_cell(self.x)



#        output_w = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[args.rnn_state_size, NOUT])) # args.rnn_state_size=400
#        output_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[NOUT]))


#        self.output = tf.nn.xw_plus_b(tf.reshape(tf.concat(self.output_list, 1), [-1, args.rnn_state_size]), 
#                                      output_w, output_b) # (batch_size, NOUT=121)

        #self.output = self.fc_output((torch.cat(self.output_list, 1).view(-1, args.rnn_state_size)))
        #self.output = self.fc_output(self.output_list.view(-1, self.args.rnn_state_size))
        self.output = self.fc_output(self.output_list.reshape(-1, self.args.rnn_state_size))

        #y1, y2, y_end_of_stroke = tf.unstack(tf.reshape(self.y, [-1, 3]), axis=1)
        y1, y2, y_end_of_stroke = torch.unbind(self.y.view(-1, 3), dim=1)


        self.end_of_stroke = 1 / (1 + torch.exp(self.output[:, 0])) # (?,), 
        pi_hat, self.mu1, self.mu2, sigma1_hat, sigma2_hat, rho_hat = torch.split(self.output[:, 1:], self.args.M, 1)
        pi_exp = torch.exp(pi_hat * (1 + self.args.b)) # args.b=3
        pi_exp_sum = torch.sum(pi_exp, 1)
        self.pi = pi_exp / expand(pi_exp_sum, 1, self.args.M)
        self.sigma1 = torch.exp(sigma1_hat - self.args.b)
        self.sigma2 = torch.exp(sigma2_hat - self.args.b)
        self.rho = torch.tanh(rho_hat)
        self.gaussian = self.pi * bivariate_gaussian(
            expand(y1, 1, self.args.M), expand(y2, 1, self.args.M),
            self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho
        )
        eps = 1e-20
        self.loss_gaussian = torch.sum(-torch.log(torch.sum(self.gaussian, 1) + eps))
        self.loss_bernoulli = torch.sum(
            -torch.log((self.end_of_stroke + eps) * y_end_of_stroke # e_t * (x_{t+1})_3 + (1 - e_t) * (1 - (x_{t+1})_3)
                    + (1 - self.end_of_stroke + eps) * (1 - y_end_of_stroke))
        )

        loss = (self.loss_gaussian + self.loss_bernoulli) / (self.args.batch_size * self.args.T)

        print('loss=', loss.cpu().item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def sample(self, length):
        x = np.zeros([1, 1, 3], np.float32)
        x[0, 0, 2] = 1 # The first point state is set to be 1
        strokes = np.zeros([length, 3], dtype=np.float32)
        strokes[0, :] = x[0, 0, :]

        #state = sess.run(self.stacked_cell.zero_state(1, tf.float32))

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
        if self.args.mode == 'synthesis':
            # print kappa_list
            import matplotlib.pyplot as plt
            plt.imshow(kappa_list, interpolation='nearest')
            plt.show()
            plt.imshow(phi_list, interpolation='nearest')
            plt.show()
            plt.imshow(w_list, interpolation='nearest')
            plt.show()
        return strokes
