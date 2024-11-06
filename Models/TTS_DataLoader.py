import torch
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HfDataset, DatasetDict, Features, Audio
from torchaudio.transforms import MelSpectrogram
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence

EOS = 'EOS'
symbols = [
    EOS, ' ', '!', ',', '-', '.', \
    ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', \
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', \
    'â', 'è', 'é', 'ê', 'ü', '’', '“', '”' \
]

symbol_to_idx = {
  s: i for i, s in enumerate(symbols)
}

def text_to_seq_char_level(text):
    text = text.lower()
    seq = []
    for symbol in text:
        idx = symbol_to_idx.get(symbol, None)
        if idx is not None:
            seq.append(idx)
    seq.append(symbol_to_idx.get(EOS))
    return torch.IntTensor(seq)



class LjSpeechDataset(Dataset):
    def __init__(self, hf_dataset: HfDataset, convert_to_mel=True, 
                 text_col='normalized_text', text_to_seq_fn=text_to_seq_char_level):
        self.hf_dataset = hf_dataset
        self.convert_to_mel = convert_to_mel
        self.text_col = text_col
        self.text_to_seq_fn = text_to_seq_fn

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx) -> Tuple[torch.IntTensor, torch.Tensor]:
        text = self.hf_dataset[idx][self.text_col]
        audio_waveform = self.hf_dataset[idx]['audio']['array']
        sampling_rate = self.hf_dataset[idx]['audio']['sampling_rate']
        
        # Apply text_to_seq_fn to the text
        text_seq = self.text_to_seq_fn(text)

        # Processing the wave-form
        mel_transform = MelSpectrogram(sampling_rate)
        mel_spec = mel_transform(audio_waveform)
        

        return text_seq, mel_spec


def speech_collate_fn(batch):
    text_seqs, mel_specs = zip(*batch)
    # collect and transpose mel_spec for pad_sequence
    mel_specs_t = [mel_spec.T for mel_spec in mel_specs]
    padded_text_seqs = pad_sequence(text_seqs, batch_first=True, padding_value=0)
    padded_mel_specs = pad_sequence(mel_specs_t, batch_first=True, padding_value=0)
    return padded_text_seqs, padded_mel_specs.permute(0, 2, 1)

def get_data_loader(dataset: HfDataset, batch_size) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=speech_collate_fn)