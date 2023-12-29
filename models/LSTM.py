# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Model(nn.Module):
    def __init__(self, configs,size=25):
        super(Model,self).__init__()
        self.task_name = configs.task_name
        self.lstm = nn.LSTM(configs.enc_in, configs.d_model, configs.e_layers, batch_first= True)
        self.mlp=nn.Linear(configs.d_model, configs.dec_in)
        self.configs = configs
        self.pred_len = configs.pred_len
    
    # def init_hidden(self, BSIZE, LEN):
    #     self.h0=torch.randn(self.configs.e_layers, self.configs.batch_size,self.configs.d_model).cuda()
    #     self.c0=torch.randn(self.configs.e_layers, self.configs.batch_size,self.configs.d_model).cuda()

    #     return (self.h0,self.c0)

    def forecast(self, x,enc_mark, dec, dec_mark):
        x=x.transpose(0,1)
        # print(x.shape)
        batch_size = x.size(0)
        # self.h0=torch.zeros(self.configs.e_layers, self.configs.batch_size,self.configs.d_model).to(device)
        # self.c0=torch.zeros(self.configs.e_layers, self.configs.batch_size,self.configs.d_model).to(device)
        self.h0=torch.zeros(self.configs.e_layers, batch_size,self.configs.d_model).to(device)
        self.c0=torch.zeros(self.configs.e_layers, batch_size,self.configs.d_model).to(device)
        output, (hn, cn) = self.lstm(x, (self.h0, self.c0))
        output = self.mlp(output)
        # if self.configs.output_attention:
        #     return output.transpose(0,1),(hn,cn)
        return output.transpose(0,1)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        pass

    def anomaly_detection(self, x_enc):
        pass

    def classification(self, x_enc, x_mark_enc):
        pass

    def forward(self, x,enc_mark, dec, dec_mark):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x,enc_mark, dec, dec_mark)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            pass
        if self.task_name == 'anomaly_detection':
            pass
        if self.task_name == 'classification':
            pass
        return None

    
  



# if __name__ == '__main__':
#     class Configs(object):
#         ab = 2
#         modes1 = 100
#         seq_len = 96*8
#         label_len = 0
#         pred_len = 720
#         output_attention = True
#         enc_in = 7
#         dec_in = 7
#         d_model = 16
#         embed = 'timeF'
#         dropout = 0.05
#         freq = 'h'
#         factor = 1
#         n_heads = 8
#         d_ff = 16
#         e_layers = 2
#         d_layers = 1
#         moving_avg = 25
#         c_out = 1
#         activation = 'gelu'
#         wavelet = 0
#         batch_size = 3

#     configs = Configs()
#     model = Model(configs).to(device)
#     LEN_TOTAL = configs.label_len + configs.seq_len
#     hidden = model.init_hidden(configs.batch_size,LEN_TOTAL)

#     enc = torch.randn([configs.batch_size, configs.seq_len, configs.enc_in])
#     enc_mark = torch.randn([configs.batch_size, configs.seq_len, configs.enc_in])

#     dec = torch.randn([configs.batch_size, configs.label_len+configs.pred_len, configs.dec_in])
#     dec_mark = torch.randn([configs.batch_size, configs.label_len+configs.pred_len, configs.dec_in])
#     out=model.forward(enc, enc_mark, dec, dec_mark)
#     print('input shape',enc.shape)
#     print('output shape',out[0].shape)
#     a = 1