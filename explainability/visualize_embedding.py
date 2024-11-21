import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from umap import UMAP
import sys
sys.path.append('../')
from model_loader import get_model
from TTS_DataLoader import symbols, text_to_seq_char_level

TRANSFORMER_MODEL = 'TransformerTTS'
SEQ2SEQ_GRU_MODEL = 'Seq2SeqTTS_GRU'

SPLIT_TRAIN = 'Train'
SPLIT_VAL = 'Validation'

MEL_BINS = 80



def plot_embeddings(embedding):
    fig, ax = plt.subplots()
    ax.scatter(embedding[:,0], embedding[:,1])
    for i, symbol in enumerate(symbols): # kmeans?
         symbol = f"({symbol})"
         ax.annotate(symbol, (embedding[i,0], embedding[i,1]), fontsize=12)
    plt.show()


def extract_vocab_embeddings(model_name:str, model: nn.Module):
    # models have different internal structure
    symbols_as_string = "".join(symbols)
    idx_string = text_to_seq_char_level(symbols_as_string)
    umap_reducer = UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=0) # 2d reduction
    print(f'All symbols in the vocabulary: {symbols_as_string}')
    print(f'As index: {idx_string}')
    if model_name == TRANSFORMER_MODEL:
        embedding = model.encoder_pre_net.enc_embedding
        # set to eval and no grad mode
        embedding.requires_grad = False
        embedding.eval()
        latent = embedding(idx_string)
        reduced = umap_reducer.fit_transform(latent.detach().cpu())
        return reduced
    pass

if __name__ == "__main__":
    print("Starting 'visualize_embedding'")
    # load model weights
    model_name = TRANSFORMER_MODEL
    split_type = SPLIT_TRAIN
    model_checkpoint_path = f'../checkpoints/{model_name}/{split_type}.pt'
    model = get_model(model_name, MEL_BINS)
    model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True))
    reduced = extract_vocab_embeddings(model_name, model)
    plot_embeddings(reduced)


