import torch
import matplotlib.pyplot as plt
from datasets import load_dataset

from model_loader import get_model
from TTS_DataLoader import text_to_seq_char_level, seq_to_text, LjSpeechDataset

# Constants
TRANSFORMER_MODEL = 'TransformerTTS'
SEQ2SEQ_GRU_MODEL = 'Seq2SeqTTS_GRU'

MEL_BINS = 80


def plot_token_contributions(contributions, tokens):
    """
    Visualize token contributions as a bar chart.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(contributions)), contributions, tick_label=tokens)
    plt.xlabel("Tokens")
    plt.ylabel("Contribution Score")
    plt.title("Token Contribution via Attention Aggregation")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def compute_token_contributions(attention_weights):
    """
    Aggregate attention weights to compute token contributions.
    Args:
        attention_weights: Tensor of shape [target_len, source_len].
    Returns:
        contributions: 1D numpy array of shape [source_len], where each value corresponds to a token's contribution.
    """
    if attention_weights.dim() == 3:
        attention_weights = attention_weights[0]  # Remove batch dimension

    contributions = attention_weights.mean(dim=0).detach().cpu().numpy()  # Shape: [source_len]
    return contributions


if __name__ == "__main__":
    print("Starting 'visualize_embedding'")

    # Load LJSpeech dataset and select one sample
    hf_dataset = load_dataset('keithito/lj_speech')['train']
    hf_dataset.set_format(type="torch", columns=["audio"], output_all_columns=True)
    lj_speech_ds = LjSpeechDataset(hf_dataset)

    # Select the first sample from the dataset
    text_seq, mel_spec = lj_speech_ds[0]
    input_text = seq_to_text(text_seq, remove_pad=True)

    # Prepare model
    model_name = TRANSFORMER_MODEL
    model_checkpoint_path = f'{model_name}/Train.pt'
    model = get_model(model_name, MEL_BINS)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    # Token processing
    input_ids = text_seq.unsqueeze(0)  # Add batch dimension (1, seq_len)
    input_tokens = list(input_text)  # Convert text sequence to tokens
    print("Input text:", input_text)
    print("Input tokens:", input_tokens)

    with torch.no_grad():
        mel_input = mel_spec.unsqueeze(0).unsqueeze(1).to('cpu')  # Add batch and channel dimensions
        text_lengths = torch.tensor([input_ids.size(1)]).to('cpu')  # Sequence lengths
        mel_lengths = torch.tensor([mel_input.size(2)]).to('cpu')  # Mel spectrogram lengths

        # Forward inference
        outputs = model.inference_forward(input_ids, text_lengths, mel_input, mel_lengths)
        _, _, _, attn_weights, _ = outputs  # Extract attention weights

    # Compute token contributions
    token_contributions = compute_token_contributions(attn_weights[0])

    # Debugging: Check token contributions
    print("Token contributions:", token_contributions)

    # Plot token contributions
    plot_token_contributions(token_contributions, input_tokens)
