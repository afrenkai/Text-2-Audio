from torch import nn
import torch

class TTS_Loss(nn.Module):
    def __init__(self):
        super(TTS_Loss, self).__init__()
        # need proper loss here
        self.mel_loss_mse_sum = torch.nn.MSELoss(reduction='sum')
        self.stop_token_loss_sum = torch.nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, mel_output: torch.Tensor, mel_target: torch.Tensor, 
                stop_token_out: torch.Tensor, stop_token_targets: torch.Tensor, mask:torch.Tensor):
            # count total number of 'real' predictions, i.e exclude masked values from loss calculations
            total_not_padding = (~mask).sum()
            n_mels_out = mel_output.size(-1)
            mel_target.requires_grad = False   
            stop_token_targets.requires_grad = False   
            mel_loss = self.mel_loss_mse_sum(mel_output, mel_target)/(n_mels_out*total_not_padding)
            stop_token_loss = self.stop_token_loss_sum(stop_token_out, stop_token_targets)/total_not_padding
            return mel_loss, stop_token_loss

class Trainer():
    def __init__(self, model : nn.Module, epochs, optimizer, 
                 criterion, train_dl, val_dl, device, 
                 checkpoint_name, teacher_f_ratio=0, grad_clip=False, max_norm=5):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.teacher_f_ratio = teacher_f_ratio
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cp_name = checkpoint_name
        self.max_epochs = epochs
        self.device = device
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.grad_clip = grad_clip
        self.max_norm = max_norm

    def train(self):
        print("Starting training")
        # trains self.model with certain params
        for epoch in range(self.max_epochs):
            self.model.train()
            running_loss = 0.0
            running_stop_loss = 0.0
            running_mel_loss = 0.0
            # train loop
            for padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets in self.train_dl:
                padded_text_seqs, padded_mel_specs = padded_text_seqs.to(self.device), padded_mel_specs.to(self.device)
                stop_token_targets = stop_token_targets.to(self.device)
                self.optimizer.zero_grad()
                mel_outputs, gate_outputs, mask = self.model(padded_text_seqs, text_seq_lens,
                                                            padded_mel_specs, mel_spec_lens, self.teacher_f_ratio)
                mel_loss, stop_token_loss = self.criterion(mel_outputs, padded_mel_specs, gate_outputs, stop_token_targets, mask)
                loss = mel_loss + stop_token_loss 
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()
                running_loss += loss.item()
                running_mel_loss += mel_loss.item()
                running_stop_loss += stop_token_loss.item()
            epoch_loss = running_loss / len(self.train_dl)
            epoch_mel_loss = running_mel_loss / len(self.train_dl)
            epoch_stop_loss = running_stop_loss / len(self.train_dl)
            # Validation
            self.model.eval()
            running_val_loss = 0.0
            running_val_stop_loss = 0.0
            running_val_mel_loss = 0.0
            with torch.no_grad():
                for padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets in self.val_dl:
                    padded_text_seqs, padded_mel_specs = padded_text_seqs.to(self.device), padded_mel_specs.to(self.device)
                    stop_token_targets = stop_token_targets.to(self.device)
                    mel_outputs, gate_outputs, mask = self.model(padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, 0)
                    mel_loss, stop_token_loss = self.criterion(mel_outputs, padded_mel_specs, gate_outputs, stop_token_targets, mask)
                    loss = mel_loss + stop_token_loss
                    running_val_loss += loss.item()
                    running_val_mel_loss += mel_loss.item()
                    running_val_stop_loss += stop_token_loss.item()
            epoch_val_loss = running_val_loss / len(self.val_dl)
            epoch_val_mel_loss = running_val_mel_loss / len(self.val_dl)
            epoch_val_stop_loss = running_val_stop_loss / len(self.val_dl)

            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                self.best_model_state_dict = self.model.state_dict()
                torch.save(self.best_model_state_dict, self.cp_name)
            print(f"Epoch {epoch + 1}/{self.max_epochs},\n"
                  f"Train Loss (total / mel / stop): {epoch_loss, epoch_mel_loss, epoch_stop_loss},\n"
                  f"Valid Loss (total / mel / stop): {epoch_val_loss, epoch_val_mel_loss, epoch_val_stop_loss}\n"
                  "------------------------------------------------------------------------------------")