from Trainer import Trainer, TTS_Loss
import TTS_DataLoader
import torch
import torch.optim as optim
import numpy as np
import os
from model_loader import get_model, get_optimizer_list

class Pipeline:
    def __init__(self, model_name, mel_bins, batch_size, lr, max_epochs, weight_decay, checkpoint_prefix, 
                 decoder_lr_ratio=2, subsample_ratio=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        self.model = get_model(model_name, mel_bins).to(device)
        optimizer_list = get_optimizer_list(self.model, lr, weight_decay, decoder_lr_ratio) 
        train_dl, val_dl, test_dl = TTS_DataLoader.load_data(batch_size, mel_bins=mel_bins, subsample_ratio=subsample_ratio)
        criterion = TTS_Loss()
        self.trainer = Trainer(self.model, max_epochs, optimizer_list, criterion,
                      train_dl, val_dl, test_dl, device, checkpoint_prefix, teacher_f_ratio=0.5, grad_clip=True, max_norm=50)

    def model_info(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        num_params = sum([np.prod(p.size()) for p in params])
        print(self.model)
        print(f'Model has {num_params} trainable parameters')
        
    def run(self):
        self.model_info()
        self.trainer.train()
        self.trainer.evaluate()


Seq2SeqTTS = 'Seq2SeqTTS'
TransformerTTS = 'TransformerTTS'

if __name__ == "__main__":
    model_name = Seq2SeqTTS  # Model name  
    # Checkpoint location
    checkpoint_dir = f'checkpoints/{model_name}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = checkpoint_dir
    # hyperparams
    mel_bins = 128
    batch_size = 16
    lr = 1e-4
    weight_decay = 1e-6
    max_epochs = 200 
    subsample_ratio = 0.2 # For testing arch

    # setup training pipeline
    Pipeline(model_name, mel_bins, batch_size, lr, max_epochs, weight_decay, checkpoint_prefix,
             subsample_ratio=subsample_ratio).run()