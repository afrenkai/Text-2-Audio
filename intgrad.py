import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from TTS_DataLoader import text_to_seq_char_level, SpeechConverter, idx_to_symbol
from model_loader import get_model
import numpy as np
from captum.attr import LayerIntegratedGradients
from datasets import load_dataset

TRANSFORMER_MODEL = 'TransformerTTS'
MEL_BINS = 80

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_token_contribution_scores(model, text_seq, text_seq_lens, mel_spec_true, mel_spec_lens):
    text_seq = text_seq.unsqueeze(0).to(device)  # shape (1, seq_len)
    text_seq_lens = torch.tensor([text_seq_lens], dtype=torch.int32).to(device)
    mel_spec_true = mel_spec_true.unsqueeze(0).to(device)  # shape (1, mel_spec_len, num_mels)
    mel_spec_lens = torch.tensor([mel_spec_lens], dtype=torch.int32).to(device)

    model.eval()
    model.to(device)

    def forward_func(input_indices, text_seq_lens, mel_spec_true, mel_spec_lens):
        outputs = model(input_indices, text_seq_lens, mel_spec_true, mel_spec_lens)
        mel_outputs = outputs[0]
        scalar_output = mel_outputs.mean(dim=(1, 2))
        return scalar_output

    lig = LayerIntegratedGradients(forward_func, model.encoder_pre_net.enc_embedding)

    attributions, delta = lig.attribute( inputs=text_seq, baselines=text_seq * 0, additional_forward_args=(text_seq_lens, mel_spec_true, mel_spec_lens), return_convergence_delta=True)
    # attributions: [batch_size, seq_len, embedding_dim]
    # Compute norm over the embed dim
    attributions = attributions.detach().cpu().numpy().squeeze(0)  # shape [seq_len, embedding_dim]
    token_contribution_scores = np.linalg.norm(attributions, axis=1)

    return token_contribution_scores, text_seq.squeeze(0).cpu()

def load_first_example():
    hf_dataset = load_dataset('keithito/lj_speech')['train']
    hf_dataset.set_format(type="torch", columns=["audio"], output_all_columns=True)
    first_example = hf_dataset[0]
    text = first_example['normalized_text']
    audio_waveform = first_example['audio']['array']
    text_seq = text_to_seq_char_level(text)
    text_seq_lens = text_seq.shape[0]

    num_mels = MEL_BINS  # 80
    speech_converter = SpeechConverter(num_mels)
    mel_spec = speech_converter.convert_to_mel_spec(audio_waveform)
    mel_spec_true = mel_spec  # shape (num_mels, mel_spec_len)
    mel_spec_lens = mel_spec_true.shape[1]  # scalar
    mel_spec_true = mel_spec_true.T  # shape (mel_spec_len, num_mels)

    return text_seq, text_seq_lens, mel_spec_true, mel_spec_lens, text

if __name__ == "__main__":
    print("Starting 'compute_token_contribution_scores'")
    model_name = TRANSFORMER_MODEL
    split_type = 'Train'
    model_checkpoint_path = f'{split_type}.pt'
    model = get_model(model_name, MEL_BINS)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)

    text_seq, text_seq_lens, mel_spec_true, mel_spec_lens, input_text = load_first_example()
    print(f"Input text: {input_text}")

    scores, idx_string = compute_token_contribution_scores(
        model, text_seq, text_seq_lens, mel_spec_true, mel_spec_lens
    )

    print("Token Contribution Scores:")
    for idx, score in enumerate(scores):
        token_idx = idx_string[idx].item()
        token = idx_to_symbol.get(token_idx, '?')
        print(f"Token: {token}, Score: {score}")

    tokens = [idx_to_symbol.get(idx.item(), '?') for idx in idx_string]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(scores)), scores, tick_label=tokens)
    plt.xlabel('Tokens')
    plt.ylabel('Contribution Score')
    plt.title('Token Contribution Scores for the First Example')
    plt.show()
