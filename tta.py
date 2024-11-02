from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextxAudioDS(Dataset):
    def __init__(self, text_data: List[str], audio_files: List[str], target_sr: int, max_text_len: int, n_mels: int):
        self.text_data = text_data
        self.audio_files = audio_files
        self.target_sr = target_sr
        self.max_text_len = max_text_len
        self.n_mels = n_mels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.mel_trans = MelSpectrogram(sample_rate = target_sr, n_mels = n_mels)

    def __len__(self) -> int:
        return len(self.text_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.text_data[idx]
        text_tokens = self.tokenizer(text, return_tensors = 'pt', padding = 'max_length', max_length = self.max_text_len)
        text_inp_ids = text_tokens['input_ids'].squeeze()
        
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(orig_freq = sample_rate, new_freq = self.target_sr)(waveform)
        mel_spectrogam = self.mel_trans(waveform).squeeze(0) #outputs [n_mels, time]
        return text_inp_ids, mel_spectrogam # as specified in spec, returns tuple of both tensors.
    
    
class TextualEncoder(nn.Module):
    
    def __init__(self, hidden_dim: int = 256):
        super(TextualEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear (self.bert.config.hidden_size, hidden_dim)
        
    def forward(self, input_ids: torch.Tensor)-> torch.Tensor:
        output = self.bert(input_ids)['last_hidden_state'] #[batch_size, sequence_len, hidden_dim]
        output = self.fc(output)
        return output
    
    
class AudioDecoder(nn.Module):
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, mel_dim: int = 80, num_heads: int = 8, num_layers: int = 3):
        super(AudioDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model = input_dim, nhead = num_heads, dim_feedforward = hidden_dim)
        self.trans_decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)
        self.fc = nn.Linear(input_dim, mel_dim)
        
    def forward(self, text_feat: torch.Tensor, target_mel_spectrogram: torch.Tensor) -> torch.Tensor:
        target_mel_spectrogram = target_mel_spectrogram.permute(2, 0, 1) #[time, batch_size, mel_dim]
        memory = text_feat.permute(1, 0, 2) # [seq_len, batch_size, hidden_dim]
        output = self.trans_decoder(target_mel_spectrogram, memory)
        mel_spec = self.fc(output)
        return mel_spec.permute(1, 2, 0) # [batch_size, mel_dim, time]
    
    
class Text2Audio(nn.Module):
    def __init__(self):
        super (Text2Audio, self).__init__()
        self.encoder = TextualEncoder()
        self.decoder = AudioDecoder()
        
    def forward(self, input_ids: torch.Tensor, target_mel_spectrogram: torch.Tensor) -> torch.Tensor:
        text_feat = self.encoder(input_ids)
        mel_spec = self.decoder(text_feat, target_mel_spectrogram)
        return mel_spec
    

def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 10
) -> None:
    model.train()
    for epoch in range(num_epochs):
        cum_loss = 0.0
        for input_ids, mel_spec  in dataloader:
            optimizer.zero_grad()
            input_ids, mel_spec = input_ids.to(device), mel_spec.to(device)
            outputs = model(input_ids, mel_spec)
            loss = criterion(outputs, mel_spec)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}") 
        
def infer(model: nn.Module, text: str) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text_tokens = tokenizer(text, return_tensors = 'pt', padding = 'max_length', max_length = 50)
        input_ids = text_tokens['input_ids'].to(device)
        mel_spec = model(input_ids, torch.zeros(1, 80, 100).to(device))
    return mel_spec
