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
        self.stacked_cell = torch.nn.LSTM(input_size=3, hidden_size=args.rnn_state_size, num_layers=2, batch_first=True)

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=args.learning_rate)

    def _expand(self, x, dim, N):
        return torch.cat([x.unsqueeze(dim) for _ in range(N)], dim)

    def _bivariate_gaussian(self, x1, x2, mu1, mu2, sigma1, sigma2, rho):
        z = torch.pow((x1 - mu1) / sigma1, 2) + torch.pow((x2 - mu2) / sigma2, 2) \
            - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
        return torch.exp(-z / (2 * (1 - torch.pow(rho, 2)))) / \
               (2 * np.pi * sigma1 * sigma2 * torch.sqrt(1 - torch.pow(rho, 2)))
   

    def fit(self, x, y):
        '''
            x: (batch_size, args.T, 3) # args.T=300 if train else 1, (batch_size, T, 3)
            y: (batch_size, args.T, 3)
        '''
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        output_list, final_state = self.stacked_cell(x)

        output = self.fc_output(output_list.reshape(-1, self.args.rnn_state_size)) # (batch_size * args.T, self.NOUT=121)

        y1, y2, y_end_of_stroke = torch.unbind(y.view(-1, 3), dim=1) # (batch_size * args.T, )


        end_of_stroke = 1 / (1 + torch.exp(output[:, 0])) # (batch_size * args.T,) 
        pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = torch.split(output[:, 1:], self.args.M, 1)
        pi_exp = torch.exp(pi_hat * (1 + self.args.b)) # args.b=3
        pi_exp_sum = torch.sum(pi_exp, 1)
        pi = pi_exp / self._expand(pi_exp_sum, 1, self.args.M)
        sigma1 = torch.exp(sigma1_hat - self.args.b)
        sigma2 = torch.exp(sigma2_hat - self.args.b)
        rho = torch.tanh(rho_hat)
        gaussian = pi * self._bivariate_gaussian(
            self._expand(y1, 1, self.args.M), self._expand(y2, 1, self.args.M),
            mu1, mu2, sigma1, sigma2, rho
        )
        eps = 1e-20
        loss_gaussian = torch.sum(-torch.log(torch.sum(gaussian, 1) + eps))
        loss_bernoulli = torch.sum(
            -torch.log((end_of_stroke + eps) * y_end_of_stroke # e_t * (x_{t+1})_3 + (1 - e_t) * (1 - (x_{t+1})_3)
                    + (1 - end_of_stroke + eps) * (1 - y_end_of_stroke))
        )

        self.loss = (loss_gaussian + loss_bernoulli) / (self.args.batch_size * self.args.T)


        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()




    def sample(self, length):
        x = np.zeros([1, 1, 3], np.float32)
        x[0, 0, 2] = 1 # The first point state is set to be 1
        strokes = np.zeros([length, 3], dtype=np.float32)
        strokes[0, :] = x[0, 0, :]

        for i in range(length - 1):
        
            output_list, final_state = self.stacked_cell(torch.Tensor(x).to(device))
            output = self.fc_output(output_list.reshape(-1, self.args.rnn_state_size))
            end_of_stroke = 1 / (1 + torch.exp(output[:, 0])) # (?,), 
            pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = torch.split(output[:, 1:], self.args.M, 1)
            pi_exp = torch.exp(pi_hat * (1 + self.args.b)) # args.b=3
            pi_exp_sum = torch.sum(pi_exp, 1)
            pi = pi_exp / self._expand(pi_exp_sum, 1, self.args.M)
            sigma1 = torch.exp(sigma1_hat - self.args.b)
            sigma2 = torch.exp(sigma2_hat - self.args.b)
            rho = torch.tanh(rho_hat)
            end_of_stroke, pi, mu1, mu2, sigma1, sigma2, rho = end_of_stroke.cpu().detach().numpy(), pi.cpu().detach().numpy(), mu1.cpu().detach().numpy(), mu2.cpu().detach().numpy(), sigma1.cpu().detach().numpy(), sigma2.cpu().detach().numpy(), rho.cpu().detach().numpy()

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
