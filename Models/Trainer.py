from torch import nn
import torch
from TtsTransformers import TTSTransformers
from tqdm import tqdm
class TTS_Loss(nn.Module):
    def __init__(self):
        super(TTS_Loss, self).__init__()
        self.mel_loss_mse = torch.nn.MSELoss()
        self.stop_token_loss_bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, mel_output, mel_target, stop_token_out, stop_token_targets):      
            mel_loss = self.mel_loss_mse(mel_output, mel_target)
            stop_token_loss = self.stop_token_loss_bce(stop_token_out, stop_token_targets)
            return mel_loss + stop_token_loss

class Trainer():
    def __init__(self, model : nn.Module, epochs, optimizer, 
                 criterion,  train_dl, val_dl, device, 
                 checkpoint_name):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cp_name = checkpoint_name
        self.max_epochs = epochs
        self.device = device
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train(self):
        print("\nStarting training")
        # trains self.model with certain params
        for epoch in range(self.max_epochs):
            self.model.train()
            running_loss = 0.0
            train_loader = tqdm(self.train_dl, desc=f"Epoch {epoch + 1}/{self.max_epochs} - Training", leave=False)
            for padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets in self.train_dl:
                padded_text_seqs, padded_mel_specs = padded_text_seqs.to(self.device), padded_mel_specs.to(self.device)
                stop_token_targets = stop_token_targets.to(self.device)
                self.optimizer.zero_grad()
                # set t-force back to 0, 0.5
                if isinstance(self.model, TTSTransformers):
                    # TTSTransformers expects 4 arguments
                    mel_outputs, gate_outputs = self.model(
                        padded_text_seqs,
                        padded_mel_specs,
                        mel_spec_lens,
                        teacher_force_ratio=0.0
                    )
                else:

                    mel_outputs, gate_outputs = self.model(
                        padded_text_seqs,
                        text_seq_lens,
                        padded_mel_specs,
                        mel_spec_lens,
                        teacher_force_ratio=0.3
                    )

                loss = self.criterion(mel_outputs, padded_mel_specs, gate_outputs, stop_token_targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                train_loader.set_postfix({"Loss": loss.item()})
                train_loader.update(1)
            epoch_loss = running_loss / len(self.train_dl)
            # Validation
            self.model.eval()
            running_val_loss = 0.0
            val_loader = tqdm(self.val_dl, desc=f"Epoch {epoch + 1}/{self.max_epochs} - Validation", leave=False)
            with torch.no_grad():
                for padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets in self.val_dl:
                    padded_text_seqs, padded_mel_specs = padded_text_seqs.to(self.device), padded_mel_specs.to(self.device)
                    stop_token_targets = stop_token_targets.to(self.device)
                    mel_outputs, gate_outputs = self.model(padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, 0)
                    loss = self.criterion(mel_outputs, padded_mel_specs, gate_outputs, stop_token_targets)
                    running_val_loss += loss.item()
                    val_loader.set_postfix({"Val Loss": loss.item()})
                    val_loader.update(1)
            epoch_val_loss = running_val_loss / len(self.val_dl)

            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                self.best_model_state_dict = self.model.state_dict()
                torch.save(self.best_model_state_dict, self.cp_name)
            print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {epoch_loss}, Val Loss: {epoch_val_loss}")




