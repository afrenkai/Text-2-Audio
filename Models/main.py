from TtsSimpleTest import TTS_Simple
from TtsTransformers import TTSTransformers
from Trainer import Trainer, TTS_Loss
import TTS_DataLoader
from datasets import load_dataset
import torch
import torch.optim as optim
import numpy as np


class Pipeline:
    def __init__(self, model_class, model_config, batch_size, lr, max_epochs, weight_decay, checkpoint_name):
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        self.model = model_class(**model_config).to(self.device)

        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.checkpoint_name = checkpoint_name

        self.loss_fn = TTS_Loss()

    def load_data(self):
        # load dataset
        hf_dataset = load_dataset('keithito/lj_speech')['train']
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
        self.train_ds = TTS_DataLoader.LjSpeechDataset(hf_train_dataset,
                                                       convert_to_mel=True,
                                                       num_mels=self.model_config['mel_bins'])
        self.val_ds = TTS_DataLoader.LjSpeechDataset(hf_val_dataset,
                                                     convert_to_mel=True,
                                                     num_mels=self.model_config['mel_bins'])
        self.test_ds = TTS_DataLoader.LjSpeechDataset(hf_test_dataset,
                                                      convert_to_mel=True,
                                                      num_mels=self.model_config['mel_bins'])
        # convert datasets to dataloader
        self.train_dl = TTS_DataLoader.get_data_loader(self.train_ds,
                                                       self.batch_size,
                                                       num_workers=4)
        self.val_dl = TTS_DataLoader.get_data_loader(self.val_ds,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     num_workers=2)
        self.test_dl = TTS_DataLoader.get_data_loader(self.test_ds,
                                                      self.batch_size, shuffle=False, num_workers=2)

    def model_info(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        num_params = sum([np.prod(p.size()) for p in params])
        print(f'Model has {num_params} trainable parameters')
        print(self.model)

    def train_model(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        trainer = Trainer(self.model,
                          self.max_epochs,
                          optimizer,
                          self.loss_fn,
                          self.train_dl,
                          self.val_dl,
                          self.device,
                          self.checkpoint_name)
        trainer.train()

    def run(self):
        self.load_data()
        self.model_info()
        self.train_model()


if __name__ == "__main__":
    common_config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'vocab_size': TTS_DataLoader.symbols_len,
        'mel_bins': 64,
        'embedding_dim': 64,
    }

    transformer_config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'vocab_size': TTS_DataLoader.symbols_len,
        'mel_bins': 64,  # Set mel_bins to match d_model for decoder
        'embedding_dim': 64,
        'n_heads_enc': 8,
        'n_heads_dec': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_ffn': 256
    }

    simple_config = {
        **common_config,
        'enc_out_size': 128,
    }

    model_classes = {
        'simple': TTS_Simple,
        'transformer': TTSTransformers
    }

    selected_model = 'transformer'
    model_class = model_classes[selected_model]
    model_config = transformer_config if selected_model == 'transformer' else simple_config
    batch_size = 32
    lr = 0.001
    max_epochs = 50
    checkpoint_name = f"{selected_model}_TtsModel.pt"
    weight_decay = 1e-4

    pipeline = Pipeline(model_class,
                        model_config,
                        batch_size,
                        lr,
                        max_epochs,
                        weight_decay,
                        checkpoint_name)
    pipeline.run()
