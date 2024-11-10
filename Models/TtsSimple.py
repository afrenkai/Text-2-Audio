import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import torchaudio.transforms as T

class TTS_Simple(nn.Module):
    def __init__(self, device, vocab_size, embedding_dim, enc_out_size, mel_bins=128):
        super(TTS_Simple, self).__init__()
        self.device = device
        # -----Encoder-----
        self.enc_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.enc_linear = nn.Linear(embedding_dim, embedding_dim)
        self.enc_conv1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1, stride=1, dilation=1)
        self.enc_batch_norm_1 = nn.BatchNorm1d(
            embedding_dim
        )
        self.enc_dropout_1 = nn.Dropout(0.5)
        self.enc_conv2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1, stride=1, dilation=1)
        self.enc_batch_norm_2 = nn.BatchNorm1d(
            embedding_dim
        )
        self.enc_dropout_2 = nn.Dropout(0.5)
        self.enc_lstm = nn.LSTM(embedding_dim, enc_out_size, batch_first=True)
        
        # -----Decoder-----
        self.dec_lstm = nn.LSTM(mel_bins, enc_out_size, batch_first=True)
        self.dec_lin_proj = nn.Linear(enc_out_size, mel_bins)
        self.dec_eos_gate = nn.Linear(enc_out_size, 1)


    def forward(self, x: torch.Tensor, text_seq_lens: torch.Tensor, y: torch.Tensor, mel_spec_lens: torch.Tensor, teacher_force_ratio=0.0):
        # ENCODER
        x = self.enc_embedding(x) # batch_size, max_seq_len, embedding_dim
        # TODO check enforce_sorted logic
        # TODO maybe keep text_seq_lens as python list
        x = self.enc_linear(x)
        x = x.permute(0, 2, 1)
        # First Conv Layer
        x = self.enc_conv1(x)
        x = self.enc_batch_norm_1(x)
        x = nn.functional.relu(x)
        x = self.enc_dropout_1(x)
        # Second Conv Layer
        x = self.enc_conv2(x)
        x = self.enc_batch_norm_2(x)
        x = nn.functional.relu(x)
        x = self.enc_dropout_2(x)
        x = x.permute(0, 2, 1)
        packed_x = pack_padded_sequence(x , text_seq_lens.cpu().numpy(), batch_first=True, enforce_sorted=False) 
        _, (hidden, cell) = self.enc_lstm(packed_x)

        # DECODER
        # init SOS for decoder input
        dec_input = self.get_decoder_sos(y) # batch, 1, n_mels
        mel_outputs = []
        gate_outputs = []
        # iterate time_dim (i.e. max_mel_length of batch)
        max_mel_len = y.size(1)
        for t_step in range(max_mel_len):
            dec_lstm_output, (hidden, cell) = self.dec_lstm(dec_input, (hidden, cell))
            # pass output of decoder lstm to fc layer to map from dec_lstm_hidden to n_mels
            dec_mel_output = self.dec_lin_proj(dec_lstm_output)
            mel_outputs.append(dec_mel_output.squeeze(1))
            # linear projection of lstm out to mel_dims
            dec_gate_output = self.dec_eos_gate(dec_lstm_output)
            gate_outputs.append(dec_gate_output.squeeze(1))
            # teacher forcing (y has shape: batch_size, mel_seq_len, n_mels)
            if torch.rand(1).item() < teacher_force_ratio:
                dec_input = y[:, t_step:t_step+1, :] # use output from actual sequence
            else:
                dec_input = dec_mel_output  # use output from pred of decoder
        mel_outputs = torch.stack(mel_outputs, dim=1)
        gate_outputs =  torch.stack(gate_outputs, dim=1).squeeze(-1)
        masked_mel_outputs, masked_gate_outputs, mask = self.mask_output(mel_outputs, gate_outputs, mel_spec_lens, max_mel_len)
        return masked_mel_outputs, masked_gate_outputs, mask

    def get_decoder_sos(self, y):
        sos =  torch.zeros(y.size(0), 1, y.size(2)).to(y.device) # batch, 1, n_mels
        return sos
    
    def mask_output(self, mel_outputs: torch.Tensor, gate_outputs: torch.Tensor, mel_spec_lens: torch.Tensor, max_mel_len):
        mask = self.get_mask(mel_spec_lens, max_mel_len)
        masked_mel_outputs = mel_outputs.masked_fill(mask.unsqueeze(-1), 0)
        masked_gate_outputs = gate_outputs.masked_fill(mask, 1e3) # Sigmoid will convert masked logit to probability ≈ 1.0   
        return masked_mel_outputs, masked_gate_outputs, mask

    def get_mask(self, mel_spec_lens, max_mel_len):
        base_mask = torch.arange(max_mel_len).expand(mel_spec_lens.size(0), max_mel_len).T
        mask = (base_mask > mel_spec_lens).to(self.device).permute(1,0).to(self.device)
        return mask



