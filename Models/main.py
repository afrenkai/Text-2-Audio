from TtsSimple import TTS_Simple
from Trainer import Trainer
import TTS_DataLoader
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


if __name__ == "__main__":  
    # setup hparams for model
    vocab_size = TTS_DataLoader.symbols_len
    mel_bins = 128 # possible hyperparameter to choose
    embedding_dim = 64
    enc_out_size = 128
    dec_lstm_out_size = 256
    
    batch_size = 16
    
    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # load dataset
    hf_dataset = load_dataset('keithito/lj_speech', trust_remote_code=True)['train']
    hf_dataset.set_format(type="torch", columns=["audio"], output_all_columns=True)

    # split dataset into training and (validation+test) set
    hf_split_datadict = hf_dataset.train_test_split(test_size=0.2)
    hf_train_dataset = hf_split_datadict['train']
    # split (validation+test) dataset into validation and test set
    hf_dataset = hf_split_datadict['test']
    hf_split_datadict = hf_dataset.train_test_split(test_size=0.5)
    hf_val_dataset = hf_split_datadict['train']
    hf_test_dataset = hf_split_datadict['test']
   
    # convert hf_dataset to pytorch datasets
    train_ds = TTS_DataLoader.LjSpeechDataset(hf_train_dataset, convert_to_mel=True, num_mels=mel_bins)
    val_ds = TTS_DataLoader.LjSpeechDataset(hf_val_dataset, convert_to_mel=True, num_mels=mel_bins)
    test_ds = TTS_DataLoader.LjSpeechDataset(hf_test_dataset, convert_to_mel=True, num_mels=mel_bins)
    # convert datasets to dataloader
    train_dl = TTS_DataLoader.get_data_loader(train_ds, batch_size, num_workers=4)
    val_dl = TTS_DataLoader.get_data_loader(val_ds, batch_size, shuffle=False, num_workers=2)
    test_dl = TTS_DataLoader.get_data_loader(test_ds, batch_size, shuffle=False, num_workers=2)

    # init model
    tts = TTS_Simple(vocab_size, embedding_dim, enc_out_size, mel_bins)
    model_parameters = filter(lambda p: p.requires_grad, tts.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Model has {num_params} trainable parameters")
    tts.to(device)


    print(tts)


    # TODO: testing for now
    padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens = next(iter(train_dl))
    padded_text_seqs, padded_mel_specs = padded_text_seqs.to(device), padded_mel_specs.to(device)
    out = tts.forward(padded_text_seqs, padded_mel_specs, 0)

    # trainer hyper params
    lr = 0.001
    loss_fn = nn.MSELoss()
    max_epochs = 40
    checkpoint_name = "TtsSimple.pt"
    weight_decay=1e-4

    

    # # setup trainer class
    trainer = Trainer(tts, max_epochs, optim.Adam(tts.parameters(), lr=lr, weight_decay=weight_decay), loss_fn,
                      train_dl, val_dl, device, checkpoint_name, log_shape=True)
    # trainer.train()