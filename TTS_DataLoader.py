import torch
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HfDataset
from torchaudio.transforms import MelSpectrogram
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
import warnings
from speech_utils import SpeechConverter

EOS = 'EOS'
symbols = [
    EOS, ' ', '!', ',', '-', '.', \
    ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', \
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', \
    'â', 'è', 'é', 'ê', 'ü', '’', '“', '”' \
]

symbols_len= len(symbols)

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
    def __init__(self, hf_dataset: HfDataset, num_mels=128,
                 text_col='normalized_text', text_to_seq_fn=text_to_seq_char_level):
        self.hf_dataset = hf_dataset
        self.num_mels = num_mels
        self.text_col = text_col
        self.text_to_seq_fn = text_to_seq_fn
        self.speech_converter = SpeechConverter(self.num_mels)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx) -> Tuple[torch.IntTensor, torch.Tensor]:
        text = self.hf_dataset[idx][self.text_col]
        audio_waveform = self.hf_dataset[idx]['audio']['array']
        # sr is constant for the dataset, use speech_utils.sr
        # sampling_rate = self.hf_dataset[idx]['audio']['sampling_rate']
        
        # Apply text_to_seq_fn to the text
        text_seq = self.text_to_seq_fn(text)

        # Processing the wave-form
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=UserWarning)
        #     mel_transform = MelSpectrogram(sampling_rate, n_mels=self.num_mels)
        # mel_spec = mel_transform(audio_waveform)
        mel_spec = self.speech_converter.convert_to_mel_spec(audio_waveform)
        
        return text_seq, mel_spec
    



def speech_collate_fn(batch):
    text_seqs, mel_specs = zip(*batch)
    text_seq_lens = [text_seq.shape[-1] for text_seq in text_seqs] # batch first
    mel_specs_t = []
    mel_spec_lens = []
    max_mel_seq = -1
    for mel_spec in mel_specs:
        mel_specs_t.append(mel_spec.T)
        true_mel_size = mel_spec.shape[-1]
        mel_spec_lens.append(true_mel_size)
        if true_mel_size > max_mel_seq:
            max_mel_seq = true_mel_size

    # need to know max size to pad to/ generate stop token input
    stop_token_targets = []
    for i in range(len(mel_specs)):
        stop_token_target = torch.zeros(max_mel_seq)
        true_mel_size = mel_spec_lens[i]
        stop_token_target[true_mel_size:] = 1
        # stop_token_target[-1] = 1 # TODO: case when true = max; should last value always be eos?
        stop_token_targets.append(stop_token_target)
    
    # pad sequence so pytorch can batch them together
    # alternatives using the minimum from the batch
    # this is using the right padding for samples that have seq_len < max_batch_seq_len   
    padded_text_seqs = pad_sequence(text_seqs, batch_first=True, padding_value=0)
    padded_mel_specs = pad_sequence(mel_specs_t, batch_first=True, padding_value=0)
    text_seq_lens = torch.IntTensor(text_seq_lens)
    mel_spec_lens = torch.IntTensor(mel_spec_lens)
    stop_token_targets = torch.stack(stop_token_targets)
    # print("In collate", padded_mel_specs.shape, stop_token_targets.shape)
    return padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets

def get_data_loader(dataset: HfDataset, batch_size, shuffle=True, num_workers=0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=speech_collate_fn, 
                      num_workers=num_workers)