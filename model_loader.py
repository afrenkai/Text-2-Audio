from models.Seq2SeqTTS import Seq2SeqTTS
from models.TtsTransformers import TTSTransformers
from TTS_DataLoader import symbols_len
from torch.optim import Adam
from torch import nn

seq2seq_config = {
            'vocab_size': symbols_len,
            'embedding_dim' : 128,
            'enc_hidden_size': 256
        }

transformer_config = {
            'vocab_size': symbols_len,
            'embedding_dim': 64,
            'n_heads_enc': 8,
            'n_heads_dec': 8,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_ffn': 256
        }

def get_model(model_name: str, mel_bins) -> nn.Module:
    model = None
    if model_name == 'Seq2SeqTTS':
        model =  Seq2SeqTTS(mel_bins=mel_bins, **seq2seq_config)
    elif model_name == 'TransformerTTS':
        model =  TTSTransformers(mel_bins=mel_bins, **transformer_config)
    else:
        raise ValueError('Invalid model name for model hyper params.')
    return model

def get_optimizer_list(model: nn.Module, lr=1e-4, weight_decay=1e-6, decoder_lr_ratio=1):
    optimizer_list = []
    if type(model) is Seq2SeqTTS:
        encoder_optimizer = Adam(model.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        decoder_optimizer = Adam(model.decoder.parameters(), lr=decoder_lr_ratio*lr, weight_decay=weight_decay)
        optimizer_list = [encoder_optimizer, decoder_optimizer]
    elif type(model) is TTSTransformers:
        optimizer_list = [Adam(model.parameters(), lr=lr, weight_decay=weight_decay)]
    else:
        raise ValueError('Invalid model name for model hyper params.')
    return optimizer_list