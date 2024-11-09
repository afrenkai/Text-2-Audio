import torch
import torch.nn as nn


# Apparently you need to add spaces between octothorpe and comments according to PEP STD who knew

class TTSTransformers(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 mel_bins,
                 n_heads_enc,
                 n_heads_dec,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_ffn):
        super(TTSTransformers, self).__init__()

        # ENCODER
        self.enc_embedding = nn.Embedding(vocab_size, embedding_dim)
        # positional encoding here
        self.positional_encoding = PositionalEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads_enc, dim_feedforward=dim_ffn)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # DECODER
        # positional encoding for decoder too
        self.mel_positional_encoding = PositionalEncoding(mel_bins)

        decoder_layer = nn.TransformerDecoderLayer(d_model=mel_bins, nhead=n_heads_dec, dim_feedforward=dim_ffn)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.decoder_fully_connected = nn.Linear(mel_bins, mel_bins)
        self.gate_layer = nn.Linear(mel_bins, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mel_spec_lens: torch.Tensor, teacher_force_ratio: float = 0.0):
        """
                Forward pass for the transformer TTS model.

                Parameters:
                    x (Tensor): Input tensor representing text tokens (batch_size, max_seq_len).
                    y (Tensor): Target tensor representing mel spectrogram frames (batch_size, mel_seq_len, mel_bins).
                    mel_spec_lens: Length tensor representing the number of spectrogram frames (i.e. [150, 200, 225] so
                    we can pad them to be the same size (or truncate)
                    teacher_force_ratio (float): Probability of using teacher forcing during training.


                Returns:
                    Tensor: Predicted mel spectrogram frames (batch_size, mel_bins, mel_seq_len).

        Basis generated with GPT.
        Prompt: Create a well-documented docstring for the forward pass of a custom transformer model,
        referred to as the transformer TTS model.
        Assume it takes in x (a tensor representing text tokens making up <batch_size, max_sequence_length>)
        and y (a target tensor representing mel spectrogram frames made up by
        <batch_size, mel_sequence_length, mel_bins>),
        as well as a teacher_force_ratio
        ( a float which determines the likelihood of using teacher forcing during training.)
                """

        # input: <batch_size, seq_len>
        # ENCODER

        x = self.enc_embedding(x)  # expected to be <batch_size, max_seq_len, embedding_dim>
        x = self.positional_encoding(x)  # time-ordering
        x = x.permute(1, 0, 2)  # tf needs <seq_len, batch_size, embedding_dim>
        enc_out = self.transformer_encoder(x)  # expected to be <max_seq_len, batch_size, enc_out_size>

        # DECODER
        # output: <batch_size, mel_seq_len, mel_bins>
        y = y.permute(1, 0, 2)  # expects (mel_seq_len, batch_size, n_mels)
        y = self.mel_positional_encoding(y)

        decoder_inp = self.get_decoder_sos(y)  # beginning of seq token
        mel_out = []
        gate_out = []

        # time series loop
        max_mel_len = y.size(1)
        for t_step in range(max_mel_len):
            decoder_inp = decoder_inp.unsqueeze(0)  # Prepare for transformer input <1, batch_size, mel_bins>
            decoder_inp = self.mel_positional_encoding(decoder_inp)
            dec_output = self.transformer_decoder(decoder_inp, enc_out)
            mel_frame = self.decoder_fully_connected(dec_output[-1])
            mel_out.append(mel_frame)
            gate_output = torch.sigmoid(self.gate_layer(mel_frame))
            mel_out.append(gate_output)

            if torch.rand(1).item() < teacher_force_ratio:
                decoder_inp = y[:, t_step, :]  # use real mel frame for next inp
            else:
                decoder_inp = mel_frame  # Use pred

        mel_outputs = torch.stack(mel_out).permute(1, 0, 2)  # <batch_size, mel_seq_len, mel_bins>
        gate_outputs = torch.stack(gate_out).squeeze(2).permute(1, 0)  # <batch_size, mel_seq_len>

        # Mask the outputs based on mel_spec_lens
        masked_mel_outputs, masked_gate_outputs = self.mask_output(mel_outputs, gate_outputs, mel_spec_lens,
                                                                   max_mel_len)
        return masked_mel_outputs, masked_gate_outputs

    @staticmethod
    def get_decoder_sos(y):
        sos = torch.zeros(y.size(0), y.size(2)).to(y.device)  # <batch_size, mel_bins>
        return sos

    def mask_output(self, mel_outputs, gate_outputs, mel_spec_lens, max_mel_len):
        mask = self.get_mask(mel_spec_lens, max_mel_len)
        masked_mel_outputs = mel_outputs.masked_fill(~mask.unsqueeze(-1), 0)
        masked_gate_outputs = gate_outputs.masked_fill(~mask, 1e3)
        return masked_mel_outputs, masked_gate_outputs

    def get_mask(self, mel_spec_lens, max_mel_len):
        base_mask = torch.arange(max_mel_len, device=self.device).expand(len(mel_spec_lens), max_mel_len).T
        mask = (base_mask < mel_spec_lens.unsqueeze(1)).to(self.device).permute(1, 0)
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * division_term)
        pos_enc[:, 1::2] = torch.cos(position * division_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_encoding', pos_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
               Adds positional encoding to the input tensor.

               Parameters:
                   x (Tensor): Input tensor to encode (seq_len, batch_size, embedding_dim).

               Returns:
                   Tensor: Positionally encoded tensor with same shape as input.

        Generated with GPT.
        Prompt: Create a well-documented docstring for the forward pass of a positional encoding method.
        Assume that X is made up of <sequence_length, batch_size, embedding_dimension>
               """
        x = x + self.pos_enc[:x.size(0), :]
        return x
