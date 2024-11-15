import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchaudio.transforms as T

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout=0.0, bidirectional=True):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.enc_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.enc_conv1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1, stride=1, dilation=1)
        self.enc_batch_norm_1 = nn.BatchNorm1d(
            embedding_dim
        )
        self.enc_dropout_1 = nn.Dropout(dropout)
        self.enc_conv2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1, stride=1, dilation=1)
        self.enc_batch_norm_2 = nn.BatchNorm1d(
            embedding_dim
        )
        self.enc_dropout_2 = nn.Dropout(dropout)
        self.enc_rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=self.bidirectional)

    def forward(self, x: torch.Tensor, text_seq_lens: torch.Tensor):
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
        packed_x = pack_padded_sequence(x , text_seq_lens.cpu().numpy(), batch_first=True) 
        output, hidden = self.enc_rnn(packed_x)
        output, _ = pad_packed_sequence(output, batch_first=True)
        if self.bidirectional:
            output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:] 
        return output, hidden

# Luong et al. Global attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        attn_energies = torch.sum(hidden * energy, dim=2)
        # Transpose max_length and batch_size dimensions
        # Return the softmax normalized probability scores (with added dimension)
        probs = nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)
        return probs

class Decoder(nn.Module)    :
    def __init__(self, hidden_size, mel_bins=128, dropout=0.2, num_layers=2, encoder_bidirectional=True):
        super(Decoder, self).__init__()
        self.mel_bins = mel_bins
        self.encoder_bidirectional = encoder_bidirectional
        self.num_layers = num_layers
        self.decoder_rnn = nn.GRU(self.mel_bins, hidden_size, num_layers=num_layers, 
                                  dropout=dropout, batch_first=True)
        
        self.attention = Attention(hidden_size)
        self.dec_lin_proj = nn.Linear(hidden_size, self.mel_bins)
        self.dec_eos_gate = nn.Linear(hidden_size, 1)
        self.concat2hidden = nn.Linear(hidden_size * 2, hidden_size)

    def step(self):
        pass
    
    '''
        memory : hidden and cell from encoder
        y : actual melspecs (only for teacher forcing)
        mel_spec_lens : actual un-padded lens of mel specs
        teacher_force_ratio : probability for teacher_forcing
    '''
    def forward(self, encoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, 
                y: torch.Tensor, mel_spec_lens: torch.Tensor, teacher_force_ratio=0.0):
        dec_input = self.get_decoder_sos(y.size(0), y.size(2)) # batch, 1, n_mels
        dec_input = dec_input.to(y.device)
        mel_outputs = []
        gate_outputs = []
        # iterate time_dim (i.e. max_mel_length of batch)
        max_mel_len = y.size(1)
        decoder_hidden = encoder_hidden[:self.num_layers]
        # use teacher forcing for the current batch?
        use_teacher_forcing = torch.rand(1).item() < teacher_force_ratio
        for t_step in range(max_mel_len):
            rnn_output, decoder_hidden = self.decoder_rnn(dec_input, decoder_hidden)
            attn_weights = self.attention(rnn_output, encoder_outputs)
            context = attn_weights.bmm(encoder_outputs)
            rnn_output = rnn_output.squeeze(0)
            context = context
            concat = torch.cat((rnn_output, context), 2)
            concat_output = torch.tanh(self.concat2hidden(concat))
            # Predict next melspec frame and stop token using attended rrn_output
            dec_mel_output = self.dec_lin_proj(concat_output)
            mel_outputs.append(dec_mel_output.squeeze(1))
            # linear projection of lstm out to mel_dims
            dec_gate_output = self.dec_eos_gate(concat_output)
            gate_outputs.append(dec_gate_output.squeeze(1))
            # teacher forcing (y has shape: batch_size, mel_seq_len, n_mels)
            if use_teacher_forcing:
                dec_input = y[:, t_step:t_step+1, :] # use output from actual sequence
            else:
                dec_input = dec_mel_output  # use output from pred of decoder
        mel_outputs = torch.stack(mel_outputs, dim=1)
        gate_outputs =  torch.stack(gate_outputs, dim=1).squeeze(-1)
        masked_mel_outputs, masked_gate_outputs, mask = self.mask_output(mel_outputs, gate_outputs, mel_spec_lens, max_mel_len)
    
        return masked_mel_outputs, masked_gate_outputs, mask

    def get_decoder_sos(self, batch_size, n_mels):
        return torch.zeros(batch_size, 1, n_mels) # batch, 1, n_mels

    def mask_output(self, mel_outputs: torch.Tensor, gate_outputs: torch.Tensor, mel_spec_lens: torch.Tensor, max_mel_len):
        mask = self.get_mask(mel_spec_lens, max_mel_len).to(mel_outputs.device)
        masked_mel_outputs = mel_outputs.masked_fill(mask.unsqueeze(-1), 0)
        masked_gate_outputs = gate_outputs.masked_fill(mask, 1e3) # Sigmoid will convert masked logit to probability ≈ 1.0   
        return masked_mel_outputs, masked_gate_outputs, mask

    def get_mask(self, mel_spec_lens, max_mel_len):
        base_mask = torch.arange(max_mel_len).expand(mel_spec_lens.size(0), max_mel_len).T
        mask = (base_mask > mel_spec_lens).permute(1,0)
        return mask


class Seq2SeqTTS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, 
                enc_hidden_size, encoder_bidirectional=True, dropout_encoder=0.0, mel_bins=128, dropout_decoder=0.0):
        super(Seq2SeqTTS, self).__init__()
        self.mel_bins = mel_bins
        self.encoder_bidirectional = encoder_bidirectional
        # -----Encoder-----
        # Embedding -> Convolutions -> LSTM
        self.encoder = Encoder(vocab_size, embedding_dim, enc_hidden_size, 
                dropout=dropout_encoder, bidirectional=encoder_bidirectional)
        
        # -----Decoder-----
        # Encoder Memory -> LSTM -> (mel_spec and stop tokens)
        self.decoder = Decoder(enc_hidden_size, mel_bins, dropout_decoder, 
                encoder_bidirectional=encoder_bidirectional)

    def forward(self, x: torch.Tensor, text_seq_lens: torch.Tensor, y: torch.Tensor, mel_spec_lens: torch.Tensor, teacher_force_ratio=0.0):
        encoder_outputs , hidden  = self.encoder(x, text_seq_lens)
        # DECODER
        masked_mel_outputs, masked_gate_outputs, mask = self.decoder(hidden, encoder_outputs, y, mel_spec_lens, teacher_force_ratio)
        return masked_mel_outputs, masked_gate_outputs, mask
   

    @torch.no_grad()
    # TODO: FIXME Update with new structure
    def inference(self, text_seq, max_mel_length=800, stop_token_thresh=0.5):
        self.train(False)
        self.eval()
        # assume text_seq has shape (1, input_seq_len)
        text_lengths = torch.IntTensor([text_seq.shape[-1]])
        # TODO: separate encoder and decoder logic into different classes
        #       to reduce repetition like below.
        x = self.enc_embedding(text_seq) # 1, max_seq_len, embedding_dim
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
        hidden = torch.cat(hidden.unbind(), dim=1).unsqueeze(0)
        cell = torch.cat(cell.unbind(), dim=1).unsqueeze(0)
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



