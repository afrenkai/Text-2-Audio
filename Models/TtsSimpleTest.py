import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import torchaudio.transforms as T

class TTS_Simple(nn.Module):
    def __init__(self, device, vocab_size, embedding_dim, enc_out_size, mel_bins):
        super(TTS_Simple, self).__init__()
        self.device = device
        # -----Encoder-----
        self.enc_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.enc_lstm = nn.LSTM(embedding_dim, enc_out_size, batch_first=True)
        
        # -----Decoder-----
        # self.dec_lstm = nn.LSTM(mel_bins, enc_out_size, batch_first=True)
        self.dec_lstm_cell = nn.LSTMCell(mel_bins, enc_out_size)
        self.dec_lin_proj = nn.Linear(enc_out_size, mel_bins)
        self.dec_eos_gate = nn.Linear(enc_out_size, 1)


    def forward(self, x: torch.Tensor, text_seq_lens: torch.Tensor, y: torch.Tensor, mel_spec_lens: torch.Tensor, teacher_force_ratio=0.0):

        # print("Forward called for TTS_Simple")
        # print("Input Layer:", x.shape) # batch_size, max_seq_len
        # print('----Encoder----')
        # ENCODER
        x = self.enc_embedding(x) # batch_size, max_seq_len, embedding_dim
        # print("Encoder Conv 2 Layer:", x.shape) # batch_size, max_seq_len, conv2_output_dim
        # save hidden and cell to pass to dec lstm
        # TODO check enforce_sorted logic
        # TODO maybe keep text_seq_lens as python list
        packed_x = pack_padded_sequence(x , text_seq_lens.cpu().numpy(), batch_first=True, enforce_sorted=False) 
        _, (hidden, cell) = self.enc_lstm(packed_x)
        hidden.squeeze_(0)
        cell.squeeze_(0)
        # print("Encoder LSTM Layer:", x.shape) # batch_size, max_seq_len, enc_out_size
        # DECODER
        # print('----Decoder----')
        # init SOS for decoder input
        dec_input = self.get_decoder_sos(y) # batch, 1, n_mels
        mel_outputs = []
        gate_outputs = []
        # iterate time_dim (i.e. max_mel_length of batch)
        max_mel_len = y.size(1)
        for t_step in range(max_mel_len):
            hidden, cell = self.dec_lstm_cell(dec_input, (hidden, cell))
            # pass output of decoder lstm to fc layer to map from dec_lstm_hidden to n_mels
            dec_lstm_output = self.dec_lin_proj(hidden)
            mel_outputs.append(dec_lstm_output)
            # linear projection of lstm out to mel_dims
            dec_cell_output = torch.sigmoid(self.dec_eos_gate(hidden))
            gate_outputs.append(dec_cell_output)            
            # teacher forcing (y has shape: batch_size, mel_seq_len, n_mels)
            if torch.rand(1).item() < teacher_force_ratio:
                dec_input = y[:, t_step:t_step+1, :] # use output from actual sequence
                dec_input = dec_input.squeeze(1)
            else:
                dec_input = dec_lstm_output  # use output from pred of decoder

        mel_outputs = torch.stack(mel_outputs).permute(1, 0, 2)
        gate_outputs =  torch.stack(gate_outputs).squeeze(2).permute(1,0)
        masked_mel_outputs, masked_gate_outputs = self.mask_output(mel_outputs, gate_outputs, mel_spec_lens, max_mel_len)
        return masked_mel_outputs, masked_gate_outputs

    def get_decoder_sos(self, y):
        sos =  torch.zeros(y.size(0), 1, y.size(2)).to(y.device) # batch, 1, n_mels
        sos.squeeze_(1)
        return sos
    
    def mask_output(self, mel_outputs, gate_outputs, mel_spec_lens, max_mel_len):
        mask = self.get_mask(mel_spec_lens, max_mel_len)
        masked_mel_outputs = mel_outputs.masked_fill(~mask.unsqueeze(-1), 0)
        masked_gate_outputs = gate_outputs.masked_fill(~mask, 1e3)
        return masked_mel_outputs, masked_gate_outputs


    def get_mask(self, mel_spec_lens, max_mel_len):
        base_mask = torch.arange(max_mel_len).expand(mel_spec_lens.size(0), max_mel_len).T
        return (base_mask < mel_spec_lens).to(self.device).permute(1,0).to(self.device)



