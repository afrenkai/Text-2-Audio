import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T

class TTS_Simple(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_out_size, mel_bins):
        super(TTS_Simple, self).__init__()
        # -----Encoder-----
        self.enc_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.enc_conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.enc_lstm = nn.LSTM(64, enc_out_size, batch_first=True)

        # -----Decoder-----
        self.dec_lstm = nn.LSTM(mel_bins, enc_out_size, batch_first=True)
        self.dec_fc = nn.Linear(enc_out_size, mel_bins)

    def forward(self, x: torch.Tensor, y: torch.Tensor, teacher_force_ratio=0.0):
        # print("Forward called for TTS_Simple")
        # print("Input Layer:", x.shape) # batch_size, max_seq_len
        # print('----Encoder----')
        # ENCODER
        x = self.enc_embedding(x) # batch_size, max_seq_len, embedding_dim
        # reshape x to have shape batch_size, embedding_dim, max_seq_len
        x = x.permute(0, 2, 1) # batch_size, embedding_dim, max_seq_len
        # print("Encoder Embedding Layer:", x.shape)
        x = self.enc_conv1(x) # batch_size, conv1_output_dim, max_seq_len
        # print("Encoder Conv 1 Layer:", x.shape)
        x = self.enc_conv2(x) # batch_size, conv2_output_dim, max_seq_len
        # reshape x to have shape batch_size, max_seq_len, conv2_output_dim
        x = x.permute(0, 2, 1)
        # print("Encoder Conv 2 Layer:", x.shape) # batch_size, max_seq_len, conv2_output_dim
        # save hidden and cell to pass to dec lstm
        x, (hidden, cell) = self.enc_lstm(x)
        # print("Encoder LSTM Layer:", x.shape) # batch_size, max_seq_len, enc_out_size
        # DECODER
        # print('----Decoder----')
        y = y.permute(0, 2, 1) # reshape y (batch_size, n_mels, mel_seq_len) -> (batch_size, mel_seq_len, n_mels)
        # init SOS for decoder input
        dec_input = torch.zeros(y.size(0), 1, y.size(2)).to(y.device) # batch, 1, n_mels
        outputs = []
       
        # teacher forcing (y has shape: batch_size, mel_seq_len, n_mels)
        for t_step in range(y.size(1)): # iterate time_dim
            dec_lstm_output, (hidden, cell) = self.dec_lstm(dec_input, (hidden, cell))
            # pass output of decoder lstm to fc layer to map from dec_lstm_hidden to n_mels
            dec_lstm_output = self.dec_fc(dec_lstm_output)
            outputs.append(dec_lstm_output)
            if torch.rand(1).item() < teacher_force_ratio:
                dec_input = y[:, t_step:t_step+1, :] # use output from actual sequence
            else:
                dec_input = dec_lstm_output  # use output from pred of decoder


        outputs = torch.cat(outputs, dim=1).permute(0,2,1)
        return outputs



