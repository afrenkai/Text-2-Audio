from TtsSimple import TTS_Simple
from Trainer import Trainer, TTS_Loss
import TTS_DataLoader
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


if __name__ == "__main__":     
    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # load dataset
    hf_dataset = load_dataset('keithito/lj_speech', trust_remote_code=True)['train']
    hf_dataset.set_format(type="torch", columns=["audio"], output_all_columns=True)


    # FOR REAL TRAINING
    # split dataset into training and (validation+test) set
    hf_split_datadict = hf_dataset.train_test_split(test_size=0.3)
    hf_train_dataset = hf_split_datadict['train']
    # split (validation+test) dataset into validation and test set
    hf_dataset = hf_split_datadict['test']
    hf_split_datadict = hf_dataset.train_test_split(test_size=0.5)
    hf_val_dataset = hf_split_datadict['train']
    hf_test_dataset = hf_split_datadict['test']

    # # FOR TESTING ARCH
    # hf_split_datadict = hf_dataset.train_test_split(test_size=0.8)
    # hf_train_dataset = hf_split_datadict['train']
    # # split (validation+test) dataset into validation and test set
    # hf_dataset = hf_split_datadict['test'].train_test_split(test_size=0.2)['train']
    # hf_split_datadict = hf_dataset.train_test_split(test_size=0.5)
    # hf_val_dataset = hf_split_datadict['train']
    # hf_test_dataset = hf_split_datadict['test']

    print(len(hf_train_dataset), len(hf_val_dataset), len(hf_test_dataset))

    # -- H PARAMS --
    # setup hparams for model
    vocab_size = TTS_DataLoader.symbols_len
    mel_bins = 128 # possible hyperparameter to choose
    embedding_dim = 256
    enc_out_size = 512
    batch_size = 128

    # convert hf_dataset to pytorch datasets
    train_ds = TTS_DataLoader.LjSpeechDataset(hf_train_dataset, num_mels=mel_bins)
    val_ds = TTS_DataLoader.LjSpeechDataset(hf_val_dataset, num_mels=mel_bins)
    test_ds = TTS_DataLoader.LjSpeechDataset(hf_test_dataset, num_mels=mel_bins)
    # convert datasets to dataloader
    train_dl = TTS_DataLoader.get_data_loader(train_ds, batch_size, num_workers=4)
    val_dl = TTS_DataLoader.get_data_loader(val_ds, batch_size, shuffle=False, num_workers=2)
    test_dl = TTS_DataLoader.get_data_loader(test_ds, batch_size, shuffle=False, num_workers=2)

    # init model
    tts = TTS_Simple(device, vocab_size, embedding_dim, enc_out_size, mel_bins)
    model_parameters = filter(lambda p: p.requires_grad, tts.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Model has {num_params} trainable parameters")
    tts.to(device)
    print(tts)

    ## Testing Single forward
    # padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets = next(iter(train_dl))
    # print(padded_mel_specs.shape)
    # padded_text_seqs, padded_mel_specs = padded_text_seqs.to(device), padded_mel_specs.to(device)
    # mel_out, stop_out, mask = tts.forward(padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, 1)
    # print("In main.py out (mel_out.shape, stop_out.shape, mask.shape) = ", mel_out.shape, stop_out.shape, mask.shape)


    # trainer hyper params
    
    loss_fn = TTS_Loss()
    # loss = loss_fn(mel_out, padded_mel_specs, stop_out, stop_token_targets, mask)
    max_epochs = 100
    checkpoint_name = "TtsSimple.pt"
    lr = 0.001
    weight_decay=1e-6

    # setup trainer class
    trainer = Trainer(tts, max_epochs, optim.Adam(tts.parameters(), lr=lr, weight_decay=weight_decay), loss_fn,
                      train_dl, val_dl, device, checkpoint_name, teacher_f_ratio=0.5)
    trainer.train()