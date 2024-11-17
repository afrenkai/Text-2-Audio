from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from enum import Enum


class TTS_Loss(nn.Module):
    def __init__(self, stop_token_loss_multiplier=5):
        super(TTS_Loss, self).__init__()
        # need proper loss here
        self.mel_loss_mse_sum = torch.nn.MSELoss(reduction='sum')
        self.stop_token_loss_sum = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.stop_token_alpha = stop_token_loss_multiplier

    def forward(self, mel_output: torch.Tensor, mel_target: torch.Tensor,
                stop_token_out: torch.Tensor, stop_token_targets: torch.Tensor, mask: torch.Tensor):
        # count total number of 'real' predictions, i.e exclude masked values from loss calculations
        total_not_padding = (~mask).sum()
        n_mels_out = mel_output.size(-1)
        mel_target.requires_grad = False
        stop_token_targets.requires_grad = False
        mel_loss = self.mel_loss_mse_sum(mel_output, mel_target) / (n_mels_out * total_not_padding)
        stop_token_loss = self.stop_token_loss_sum(stop_token_out, stop_token_targets) / total_not_padding
        return mel_loss, stop_token_loss * self.stop_token_alpha


class LossType(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Trainer():
    def __init__(self, model, epochs, optimizers: list,
                 criterion, train_dl, val_dl, test_dl, device,
                 checkpoint_prefix, teacher_f_ratio=0, grad_clip=False, max_norm=5):
        self.model = model
        self.optimizers = optimizers  # List of Optimizers [op1, op2]
        self.criterion = criterion
        self.teacher_f_ratio = teacher_f_ratio
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl  # TODO: write a test function for this
        self.checkpoint_prefix = checkpoint_prefix  # prefix for checkpoint
        self.max_epochs = epochs
        self.device = device
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_model_state = None
        self.grad_clip = grad_clip
        self.max_norm = max_norm
        self.writer = SummaryWriter(f'logs/{model.__class__.__name__}')

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
                padded_text_seqs = padded_text_seqs.to(self.device)
                padded_mel_specs = padded_mel_specs.to(self.device)
                stop_token_targets = stop_token_targets.to(self.device)
                for op in self.optimizers:
                    op.zero_grad()

                # Adjust the call based on the model type
                if hasattr(self.model, 'teacher_forcing_ratio'):
                    # If the model uses teacher forcing, pass the ratio
                    mel_outputs, gate_outputs, mask = self.model(
                        padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, self.teacher_f_ratio
                    )
                else:
                    # For models that don't use teacher forcing
                    mel_outputs, gate_outputs, mask = self.model(
                        padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens
                    )

                mel_loss, stop_token_loss = self.criterion(
                    mel_outputs, padded_mel_specs, gate_outputs, stop_token_targets, mask
                )
                loss = mel_loss + stop_token_loss
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                for op in self.optimizers:
                    op.step()
                running_loss += loss.item()
                running_mel_loss += mel_loss.item()
                running_stop_loss += stop_token_loss.item()
            epoch_loss = running_loss / len(self.train_dl)
            epoch_mel_loss = running_mel_loss / len(self.train_dl)
            epoch_stop_loss = running_stop_loss / len(self.train_dl)
            if epoch_loss < self.best_train_loss:
                self.best_train_loss = epoch_loss
                # For train, replace with the state of the dataloader, optimizers, model, epoch number
                torch.save(self.model.state_dict(), self.checkpoint_prefix + "_Train.pt")
            # Validation
            self.model.eval()
            running_val_loss = 0.0
            running_val_stop_loss = 0.0
            running_val_mel_loss = 0.0
            with torch.no_grad():
                for padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets in self.val_dl:
                    padded_text_seqs = padded_text_seqs.to(self.device)
                    padded_mel_specs = padded_mel_specs.to(self.device)
                    stop_token_targets = stop_token_targets.to(self.device)

                    if hasattr(self.model, 'teacher_forcing_ratio'):
                        mel_outputs, gate_outputs, mask = self.model(
                            padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, 0
                        )
                    else:
                        mel_outputs, gate_outputs, mask = self.model(
                            padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens
                        )

                    mel_loss, stop_token_loss = self.criterion(
                        mel_outputs, padded_mel_specs, gate_outputs, stop_token_targets, mask
                    )
                    loss = mel_loss + stop_token_loss
                    running_val_loss += loss.item()
                    running_val_mel_loss += mel_loss.item()
                    running_val_stop_loss += stop_token_loss.item()
            epoch_val_loss = running_val_loss / len(self.val_dl)
            epoch_val_mel_loss = running_val_mel_loss / len(self.val_dl)
            epoch_val_stop_loss = running_val_stop_loss / len(self.val_dl)

            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                torch.save(self.model.state_dict(), self.checkpoint_prefix + "_Validation.pt")

            # Log step to TensorBoard
            step_number = epoch * len(self.train_dl) + 1
            self.log_losses(epoch_loss, epoch_mel_loss, epoch_stop_loss, step_number, LossType.TRAIN)
            self.log_losses(epoch_val_loss, epoch_val_mel_loss, epoch_val_stop_loss, step_number, LossType.VAL)
            print(f"Epoch {epoch + 1}/{self.max_epochs},\n"
                  f"Train Loss (total / mel / stop): {epoch_loss, epoch_mel_loss, epoch_stop_loss},\n"
                  f"Valid Loss (total / mel / stop): {epoch_val_loss, epoch_val_mel_loss, epoch_val_stop_loss}\n"
                  "------------------------------------------------------------------------------------")
        self.writer.close()

    def log_losses(self, epoch_loss, epoch_mel_loss, epoch_stop_loss, step_number, loss_type: LossType):
        scaler_name = 'Default'
        if loss_type == LossType.TRAIN:
            scaler_name = 'Train Loss'
        elif loss_type == LossType.VAL:
            scaler_name = 'Val Loss'
        elif loss_type == LossType.TEST:
            scaler_name = 'Test Loss'
        self.writer.add_scalar(f'{scaler_name} (Total)', epoch_loss, step_number)
        self.writer.add_scalar(f'{scaler_name} (Mel)', epoch_mel_loss, step_number)
        self.writer.add_scalar(f'{scaler_name} (Stop-Token)', epoch_stop_loss, step_number)

    # function to compute test loss
    @torch.no_grad()
    def evaluate_on_test(self):
        running_loss = 0.0
        running_mel_loss = 0.0
        running_stop_loss = 0.0
        for padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets in self.test_dl:
            padded_text_seqs = padded_text_seqs.to(self.device)
            padded_mel_specs = padded_mel_specs.to(self.device)
            stop_token_targets = stop_token_targets.to(self.device)

            if hasattr(self.model, 'teacher_forcing_ratio'):
                mel_outputs, gate_outputs, mask = self.model(
                    padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, 0
                )
            else:
                mel_outputs, gate_outputs, mask = self.model(
                    padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens
                )

            mel_loss, stop_token_loss = self.criterion(
                mel_outputs, padded_mel_specs, gate_outputs, stop_token_targets, mask
            )
            loss = mel_loss + stop_token_loss
            running_loss += loss.item()
            running_mel_loss += mel_loss.item()
            running_stop_loss += stop_token_loss.item()
        loss = running_loss / len(self.test_dl)
        mel_loss = running_mel_loss / len(self.test_dl)
        stop_loss = running_stop_loss / len(self.test_dl)
        print(f"Test Loss: (total / mel / stop): {loss, mel_loss, stop_loss}")
