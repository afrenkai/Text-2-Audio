from Models.TtsSimple import TTS_Simple
from Trainer import Trainer, TTS_Loss
import TTS_DataLoader
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


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

    print('Train, ', 'Validation, ', 'Test')
    print(len(hf_train_dataset), len(hf_val_dataset), len(hf_test_dataset))

    # -- H PARAMS --
    # setup hparams for model
    vocab_size = TTS_DataLoader.symbols_len
    mel_bins = 96 # possible hyperparameter to choose
    embedding_dim = 128
    enc_hidden_size = 256

    # BATCH SIZE
    batch_size = 32

    # convert hf_dataset to pytorch datasets
    train_ds = TTS_DataLoader.LjSpeechDataset(hf_train_dataset, num_mels=mel_bins)
    val_ds = TTS_DataLoader.LjSpeechDataset(hf_val_dataset, num_mels=mel_bins)
    test_ds = TTS_DataLoader.LjSpeechDataset(hf_test_dataset, num_mels=mel_bins)
    # convert datasets to dataloader
    train_dl = TTS_DataLoader.get_data_loader(train_ds, batch_size, num_workers=2)
    val_dl = TTS_DataLoader.get_data_loader(val_ds, batch_size, shuffle=False, num_workers=1)
    test_dl = TTS_DataLoader.get_data_loader(test_ds, batch_size, shuffle=False, num_workers=1)

    # init model
    tts = TTS_Simple(device, vocab_size, embedding_dim, enc_hidden_size, 
                     encoder_bidirectional=True, mel_bins=mel_bins, dropout_encoder=0.5)
    model_parameters = filter(lambda p: p.requires_grad, tts.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Model has {num_params} trainable parameters")
    tts.to(device)
    print(tts)

    # # Testing forward for a single batch
    # padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets = next(iter(train_dl))
    # padded_text_seqs, padded_mel_specs = padded_text_seqs.to(device), padded_mel_specs.to(device)
    # mel_out, stop_out, mask = tts.forward(padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, 1)
    # print("In main.py out (mel_out.shape, stop_out.shape, mask.shape) = ", mel_out.shape, stop_out.shape, mask.shape)


    # Trainer hyper params
    loss_fn = TTS_Loss()
    max_epochs = 200
    lr = 1e-3
    decoder_lr_multiplier = 1
    weight_decay=1e-6
    encoder_optimizer = optim.Adam(tts.encoder.parameters(), lr=lr, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(tts.decoder.parameters(), lr=decoder_lr_multiplier*lr,
                                   weight_decay=weight_decay)
    optimizers = [encoder_optimizer, decoder_optimizer]
    # Checkpoint location
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_name = checkpoint_dir+"/TtsSimple"
    # setup trainer class
    trainer = Trainer(tts, max_epochs, optimizers, loss_fn,
                      train_dl, val_dl, device, checkpoint_name, teacher_f_ratio=0.5, grad_clip=True, max_norm=50)
    trainer.train()