from src.LSTM_AUTOENCODER_PLOT_FUNCTIONS  import *
import copy
import time
import pickle
import numpy as np
import torch
from torch import nn

def convert_to_tensor(data):
    sequences = data.tolist()
    #dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    dataset = [torch.tensor(s).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features

def predict(model, dataset):
    predictions, losses = [], []
    #criterion = nn.L1Loss(reduction='sum').to(model.device)
    criterion = nn.MSELoss().to(model.device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(model.device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses

def check_device():
    """
    #for Google colab
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.utils.utils as xu

    device = xm.xla_device()
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    """
    
    print(f'(Using device: {device}) ', end=" ")
    return device

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, encoding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.hidden_dim = 2 * encoding_dim
        
        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.encoding_dim,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        #return hidden_n.reshape((self.n_features, self.encoding_dim))
        return hidden_n.reshape((1, self.encoding_dim))

class Decoder(nn.Module):
    
    def __init__(self, seq_len, encoding_dim=64, n_features=1):
        super(Decoder, self).__init__()
    
        self.seq_len = seq_len
        self.encoding_dim = encoding_dim
        self.hidden_dim = 2 * encoding_dim
        self.n_features = n_features
    
        self.rnn1 = nn.LSTM(
          input_size=self.encoding_dim,
          hidden_size=self.encoding_dim,
          num_layers=1,
          batch_first=True
        )
    
        self.rnn2 = nn.LSTM(
          input_size=self.encoding_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
    
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
    
    def forward(self, x):
        #x = x.repeat(self.seq_len, self.n_features)
        #x = x.reshape((self.n_features, self.seq_len, self.encoding_dim))
        x = x.repeat(self.seq_len, 1)
        x = x.reshape((1, self.seq_len, self.encoding_dim))
        
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)

class Autoencoder(nn.Module):
    
    def __init__(self, seq_len, n_features, encoding_dim=64):
        super(Autoencoder, self).__init__()
        self.device = check_device()
        self.encoder = Encoder(seq_len, n_features, encoding_dim).to(self.device)
        self.decoder = Decoder(seq_len, encoding_dim, n_features).to(self.device)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def train_model(self, model, train_dataset, val_dataset,train_index=None, test_index=None, N_EPOCHS=0, LEARNING_RATE=1e-3, MODEL_NAME='model', plot_result=True):
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        #criterion = nn.L1Loss(reduction='sum').to(self.device)
        criterion = nn.MSELoss().to(self.device)
        history = dict(train=[], val=[], train_index=train_index, test_index=test_index)
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 100000000000000.0
      
      
        for epoch in range(1, N_EPOCHS + 1):
            
            train_losses, val_losses = [], []
            
            begin_time = time.strftime("%H:%M:%S", time.localtime())
            print(f'Time: {begin_time} -- Training Epoch {epoch}...', end=" ")
            
            
            model = model.train()
            
            for seq_true in train_dataset:
                optimizer.zero_grad()
                seq_true = seq_true.to(self.device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
      
            model = model.eval()
            
            with torch.no_grad():
                for seq_true in val_dataset:
                    seq_true = seq_true.to(self.device)
                    seq_pred = model(seq_true)
                    loss = criterion(seq_pred, seq_true)
                    val_losses.append(loss.item())
  
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
    
            history['train'].append(train_loss)
            history['val'].append(val_loss)
  
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, str(MODEL_NAME+'.pth'))
                with open(str(MODEL_NAME+'.pickle'), 'wb') as handle: pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if plot_result: plot_loss_throw_epochs(history,MODEL_NAME)
            
            end_time = time.strftime("%H:%M:%S", time.localtime())
            print(f'Time: {end_time} -- train_loss: {round(train_loss, 6)}, val_loss: {round(val_loss, 6)}')
            
            
        model.load_state_dict(best_model_wts)
        
        return model.eval(), history

  



