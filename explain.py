import torch
import torch.nn as nn
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from captum.attr import IntegratedGradients

def visualize_attention(attention_weights: torch.Tensor, text_tokens : List[str], title: str = "Attemntion Weights") -> None:
    plt.figure(figsize=(10,10))

    sns.heatmap(attention_weights.detach.cpu().numpy(), xticklabels=text_tokens, yticklabels=False, cmap='viridis', annot=True)
    plt.title(title)
    plt.xlabel('Text Tokens')
    plt.ylabel('Mel Spectrogram Frames')
    plt.show()

def compute_saliency(model: nn.Module, input_ids: torch.Tensor, target_frame_idx : int) -> np.ndarray:
    input_ids.requires_grad = True
    output = model(input_ids)
    target_frame = output[0, : , target_frame_idx]
    target_frame.mean().backward()
    saliency = input_ids.grad.abs().sum(dim=1).squeeze().detach().cpu().numpy()
    return saliency

def integrated_gradients(model: nn.Module, input_ids: torch.Tensor, target_frame_idx: int) -> np.ndarray:
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(input_ids, target_frame_idx, return_convergence_delta=True)
    return attributions.squeeze().detach().cpu().numpy()

def token_contribution_score(
        model: nn.Module,
        input_ids: torch.Tensor,
        original_text: str,
        target_frame_idx: int,
        mask_token_id: int
) -> List[float]:
    contributions = []
    for i in range(input_ids.size(1)):
        masked_inp = input_ids.clone()
        masked_inp[0, i] = mask_token_id
        masked_output = model(masked_inp)
        diff = (original_text - masked_output).abs().sum()

        contributions.append(diff.item())
    return contributions
