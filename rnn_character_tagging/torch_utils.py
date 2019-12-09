import numpy as np
import torch
import torch.nn  as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable


class TimeDistributed(nn.Module):
    def __init__(self, module):
        """
        from here: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py

        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        :param module: Module to apply input to.
        """
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x)
        x = x.contiguous().view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class AverageMeter(object):
    """
    from here: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py

    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accuracy(nn.Module):
    """
    wrapper to compute accuracy
    """
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self,y_pred,y):
        y_pred = (y_pred.view(-1, 1) > 0.5).data.float()
        y = y.view(-1, 1).data.float()
        # if using previous torch versions
        # acc = (y_pred == y).sum()/y.size(0)
        acc = (y_pred == y).sum().item()/y.size(0)
        return Variable(torch.FloatTensor([acc]))

    def __repr__(self):
        return self.__class__.__name__ + '(\n)'


class RNNCharTagger(nn.Module):
    def __init__(self, lstm_layers, input_dim, out_dim, batch_size, dropout, batch_first=True):
        super(RNNCharTagger, self).__init__()

        self.lstm_layers = lstm_layers
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.batch_first = batch_first
        self.batch_size = batch_size

        self.lstm1 =  nn.LSTM(self.input_dim, self.out_dim, batch_first=self.batch_first)
        self.drop1 = nn.Dropout(self.dropout)
        for i in range(1,self.lstm_layers):
        	if (i+1) < self.lstm_layers:
	            setattr(self, 'lstm'+str(i+1), nn.LSTM(self.out_dim, self.out_dim, batch_first=self.batch_first))
	            setattr(self, 'drop'+str(i+1), nn.Dropout(self.dropout))
	        else:
	            setattr(self, 'lstm'+str(i+1), nn.LSTM(self.out_dim, self.out_dim, batch_first=self.batch_first))
        self.linear = TimeDistributed(nn.Linear(self.out_dim, 1))

        for i in range(self.lstm_layers):

            # one could also initialize as zeros
            # setattr(self, 'h'+str(i+1), nn.Parameter(torch.zeros(1, self.batch_size, self.out_dim)))
            # setattr(self, 'c'+str(i+1), nn.Parameter(torch.zeros(1, self.batch_size, self.out_dim)))
            setattr(self, 'h'+str(i+1), nn.Parameter(nn.init.normal_(torch.Tensor(1, self.batch_size, self.out_dim))))
            setattr(self, 'c'+str(i+1), nn.Parameter(nn.init.normal_(torch.Tensor(1, self.batch_size, self.out_dim))))

    def forward(self, X):

        output, (h1, c1) = self.lstm1(X, (self.h1, self.c1))
        output = self.drop1(output)
        hidden_states = [(h1,c1)]
        for i in range(1,self.lstm_layers):
            h,c = getattr(self, 'h'+str(i+1)), getattr(self, 'c'+str(i+1))
            output, (nh,nc) = getattr(self, 'lstm'+str(i+1))(output, (h,c))
            if (i+1) < self.lstm_layers:
	            output = getattr(self, 'drop'+str(i+1))(output)
            hidden_states.append((nh,nc))

        for i in range(self.lstm_layers):
            setattr(self, 'h'+str(i+1), nn.Parameter(hidden_states[i][0].data))
            setattr(self, 'c'+str(i+1), nn.Parameter(hidden_states[i][1].data))

        output = torch.sigmoid(self.linear(output))

        return output


class BiRNNCharTagger(nn.Module):
    def __init__(self, lstm_layers, input_dim, out_dim, batch_size, dropout, batch_first=True):
        super(BiRNNCharTagger, self).__init__()

        self.lstm_layers = lstm_layers
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.batch_first = batch_first
        self.batch_size = batch_size

        self.lstm =  nn.LSTM(
            self.input_dim,
            self.out_dim,
            batch_first=self.batch_first,
            dropout=self.dropout,
            num_layers = self.lstm_layers,
            bidirectional=True)
        self.linear = TimeDistributed(nn.Linear(2*self.out_dim, 1))

    def forward(self, X):
        lstm_output, hidden = self.lstm(X)
        output = torch.sigmoid(self.linear(lstm_output))
        return output
