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
        self.mel_bins = mel_bins
        # -----Encoder-----
        self.enc_embedding = nn.Embedding(vocab_size, embedding_dim)
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
        self.enc_lstm = nn.LSTM(embedding_dim, enc_out_size, batch_first=True, bidirectional=True)
        
        # -----Decoder-----
        self.dec_lstm = nn.LSTM(self.mel_bins, 2*enc_out_size, batch_first=True)
        self.dec_lin_proj = nn.Linear(2*enc_out_size, self.mel_bins)
        self.dec_eos_gate = nn.Linear(2*enc_out_size, 1)


    def forward(self, x: torch.Tensor, text_seq_lens: torch.Tensor, y: torch.Tensor, mel_spec_lens: torch.Tensor, teacher_force_ratio=0.0):
        # ENCODER
        x = self.enc_embedding(x) # batch_size, max_seq_len, embedding_dim
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
        # reshape encoder hidden states: (num_directions, batch, hidden) -> (1, batch, num_dir*hidden)
        hidden = torch.cat(hidden.unbind(), dim=1).unsqueeze(0)
        cell = torch.cat(cell.unbind(), dim=1).unsqueeze(0)

        # DECODER
        # init SOS for decoder input
        dec_input = self.get_decoder_sos(y.size(0), y.size(2)) # batch, 1, n_mels
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
   
    def get_decoder_sos(self, batch_size, n_mels):
        return torch.zeros(batch_size, 1, n_mels).to(self.device) # batch, 1, n_mels

    
    def mask_output(self, mel_outputs: torch.Tensor, gate_outputs: torch.Tensor, mel_spec_lens: torch.Tensor, max_mel_len):
        mask = self.get_mask(mel_spec_lens, max_mel_len)
        masked_mel_outputs = mel_outputs.masked_fill(mask.unsqueeze(-1), 0)
        masked_gate_outputs = gate_outputs.masked_fill(mask, 1e3) # Sigmoid will convert masked logit to probability ≈ 1.0   
        return masked_mel_outputs, masked_gate_outputs, mask

    def get_mask(self, mel_spec_lens, max_mel_len):
        base_mask = torch.arange(max_mel_len).expand(mel_spec_lens.size(0), max_mel_len).T
        mask = (base_mask > mel_spec_lens).to(self.device).permute(1,0).to(self.device)
        return mask

    @torch.no_grad()
    def inference(self, text_seq, max_mel_length=800, stop_token_thresh=0.5):
        self.train(False)
        self.eval()
        # assume text_seq has shape (1, input_seq_len)
        text_lengths = torch.IntTensor([text_seq.shape[-1]])
        # TODO: separate encoder and decoder logic into different classes
        #       to reduce repetition like below.
        x = self.enc_embedding(text_seq) # 1, max_seq_len, embedding_dim
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
        packed_x = pack_padded_sequence(x , text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False) 
        _, (hidden, cell) = self.enc_lstm(packed_x)
        inference_batch_size = 1
        dec_input = self.get_decoder_sos(inference_batch_size, self.mel_bins) # (1, mel_bins)
        mel_outputs = []
        for t_step in range(max_mel_length):
            dec_lstm_output, (hidden, cell) = self.dec_lstm(dec_input, (hidden, cell))
            # pass output of decoder lstm to fc layer to map from dec_lstm_hidden to n_mels
            dec_mel_output = self.dec_lin_proj(dec_lstm_output)
            mel_outputs.append(dec_mel_output.squeeze(1))
            # linear projection of lstm out to mel_dims
            dec_gate_output = self.dec_eos_gate(dec_lstm_output)
            # teacher forcing (y has shape: batch_size, mel_seq_len, n_mels)
            # if network predicted stop token
            if torch.nn.functional.sigmoid(dec_gate_output) > stop_token_thresh:
                break
            if t_step == max_mel_length-1:
                print('WARNING: max_mel_length (800) reached, model wanted to generate longer speech.')

        mel_outputs = torch.stack(mel_outputs, dim=1).to(self.device)
        return mel_outputs



