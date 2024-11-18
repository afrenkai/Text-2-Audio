from Models.Seq2SeqTTS import Seq2SeqTTS
from Models.TtsTransformers import TTSTransformers
from Models.SimpleTransformer import SimpleTTS
from TTS_DataLoader import symbols_len
from torch.optim import Adam
from torch import nn

seq2seq_config = {
            'vocab_size': symbols_len,
            'embedding_dim' : 256,
            'enc_hidden_size': 128,
            'decoder_hidden_size' : 512,
            'pre_net_out_size' : 256,
            'location_n_filters' : 32,
            'location_kernel_size' : 31,
            'attn_dim' : 128,
            'dropout_encoder' : 0.2,
            'dropout_decoder' : 0.1,
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
simple_config = {
    'text_num_embeddings': 2 * symbols_len,
    'encoder_embedding_size': 256,
    'dim_feedforward': 1024,
    'embedding_size' : 256,
    'postnet_kernel_size' : 5,
    'postnet_embedding_size' : 1024,
    'encoder_kernel_size' : 3,

}

def get_model(model_name: str, mel_bins) -> nn.Module:
    model = None
    if model_name == 'Seq2SeqTTS':
        model =  Seq2SeqTTS(mel_bins=mel_bins, **seq2seq_config)
    elif model_name == 'TransformerTTS':
        model =  TTSTransformers(mel_bins=mel_bins, **transformer_config)
    elif model_name == 'SimpleTTS':
        model = SimpleTTS(mel_bins = mel_bins, **simple_config)
    else:
        raise ValueError('Invalid model name for model hyper params.')
    return model

# return optimizer for specific model
def get_optimizer(model: nn.Module, lr=1e-4, weight_decay=1e-6):
    optimizer = None
    if type(model) is Seq2SeqTTS:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif type(model) is TTSTransformers:
        optimizer_list = [Adam(model.parameters(), lr=lr, weight_decay=weight_decay)]
    elif type(model) is SimpleTTS:
        optimizer_list = [Adam(model.parameters(), lr=lr, weight_decay=weight_decay)]
    else:
        raise ValueError('Invalid model name for model hyper params.')
    return optimizer