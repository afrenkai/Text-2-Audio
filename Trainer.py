from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from enum import Enum
from logging_utils import alignment_to_numpy, spectrogram_to_numpy
from speech_utils import SpeechConverter
from TTS_DataLoader import seq_to_text


class TTS_Loss(nn.Module):
    def __init__(self, stop_token_loss_multiplier=5):
        super(TTS_Loss, self).__init__()
        self.mel_loss_mse_sum = torch.nn.MSELoss(reduction='sum')
        self.post_net_mse_sum = torch.nn.MSELoss(reduction='sum')
        self.stop_token_loss_sum = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.stop_token_alpha = stop_token_loss_multiplier

    def forward(self, mel_output: torch.Tensor, post_net_out: torch.Tensor, mel_target: torch.Tensor,
                stop_token_out: torch.Tensor, stop_token_targets: torch.Tensor, mask: torch.Tensor):
        # count total number of 'real' predictions, i.e exclude masked values from loss calculations
        total_not_padding = (~mask).sum()
        n_mels_out = mel_output.size(-1)
        mel_target.requires_grad = False
        stop_token_targets.requires_grad = False
        mel_loss = self.mel_loss_mse_sum(mel_output, mel_target) / (n_mels_out * total_not_padding)
        post_net_loss = self.mel_loss_mse_sum(post_net_out, mel_target) / (n_mels_out * total_not_padding)
        stop_token_loss = self.stop_token_loss_sum(stop_token_out, stop_token_targets) / total_not_padding
        return mel_loss+post_net_loss, stop_token_loss * self.stop_token_alpha


class LossType(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Trainer():
    def __init__(self, mel_bins, model, epochs, optimizer, 
                 criterion, train_dl, val_dl, test_dl, device, 
                 checkpoint_prefix, teacher_f_ratio=0, grad_clip=False, max_norm=1):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.teacher_f_ratio = teacher_f_ratio
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.checkpoint_prefix = checkpoint_prefix # prefix for checkpoint 
        self.max_epochs = epochs
        self.device = device
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_model_state = None
        self.grad_clip = grad_clip
        self.max_norm = max_norm
        self.writer = SummaryWriter(f'logs/{model.__class__.__name__}')
        self.sc = SpeechConverter(mel_bins)

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
                self.optimizer.zero_grad()

                # Adjust the call based on the model type
                if hasattr(self.model, 'teacher_forcing_ratio'):
                    # If the model uses teacher forcing, pass the ratio
                    mel_outputs, post_net_outputs, gate_outputs, attention_outputs, mask = self.model(
                        padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, self.teacher_f_ratio
                    )
                else:
                    # For models that don't use teacher forcing
                    mel_outputs, post_net_outputs, gate_outputs, attention_outputs, mask  = self.model(
                        padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens
                    )

                mel_loss, stop_token_loss = self.criterion(mel_outputs, post_net_outputs, padded_mel_specs, 
                                                           gate_outputs, stop_token_targets, mask)
                    
                loss = mel_loss + stop_token_loss
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()
                running_loss += loss.item()
                running_mel_loss += mel_loss.item()
                running_stop_loss += stop_token_loss.item()
                sample_mel_train = post_net_outputs[0]
                sample_text_train = padded_text_seqs[0]
                sample_alignment_train = attention_outputs[0]
            epoch_loss = running_loss / len(self.train_dl)
            epoch_mel_loss = running_mel_loss / len(self.train_dl)
            epoch_stop_loss = running_stop_loss / len(self.train_dl)
            if epoch_loss < self.best_train_loss:
                self.best_train_loss = epoch_loss
                # TODO save dataloader, epoch number, optimizer states as well
                torch.save(self.model.state_dict(), self.checkpoint_prefix+"/Train.pt")
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
                        mel_outputs, post_net_outputs, gate_outputs, attention_outputs, mask  = self.model(
                            padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, 0
                        )
                    else:
                        mel_outputs, post_net_outputs, gate_outputs, attention_outputs, mask = self.model(
                            padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens
                        )

                    mel_loss, stop_token_loss = self.criterion(
                        mel_outputs, post_net_outputs, padded_mel_specs, gate_outputs, stop_token_targets, mask
                    )

                    loss = mel_loss + stop_token_loss
                    running_val_loss += loss.item()
                    running_val_mel_loss += mel_loss.item()
                    running_val_stop_loss += stop_token_loss.item()
                    sample_mel_val = post_net_outputs[0]
                    sample_text_val= padded_text_seqs[0]
                    sample_alignment_val = attention_outputs[0]
            epoch_val_loss = running_val_loss / len(self.val_dl)
            epoch_val_mel_loss = running_val_mel_loss / len(self.val_dl)
            epoch_val_stop_loss = running_val_stop_loss / len(self.val_dl)

            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                torch.save(self.model.state_dict(), self.checkpoint_prefix+"/Validation.pt")
            # Log step to TensorBoard
            step_number = epoch * len(self.train_dl) + 1
            self.log_losses(epoch_loss, epoch_mel_loss, epoch_stop_loss, step_number, LossType.TRAIN)
            self.log_losses(epoch_val_loss, epoch_val_mel_loss, epoch_val_stop_loss, step_number, LossType.VAL)
            self.log_images(sample_mel_train, sample_alignment_train, step_number, LossType.TRAIN)
            self.log_images(sample_mel_val, sample_alignment_val, step_number, LossType.VAL)
            self.log_sound(sample_mel_train, sample_text_train, step_number, LossType.TRAIN)
            self.log_sound(sample_mel_val, sample_text_val, step_number, LossType.VAL)
            print(f"Epoch {epoch + 1}/{self.max_epochs},\n"
                  f"Train Loss (total / mel / stop): {epoch_loss, epoch_mel_loss, epoch_stop_loss},\n"
                  f"Valid Loss (total / mel / stop): {epoch_val_loss, epoch_val_mel_loss, epoch_val_stop_loss}\n"
                  "------------------------------------------------------------------------------------")
        self.writer.close()

    @staticmethod
    def get_log_name(loss_type: LossType):
        name = 'Default'
        if loss_type == LossType.TRAIN:
            name = 'Train'
        elif loss_type == LossType.VAL:
            name = 'Val'
        elif loss_type == LossType.TEST:
            name = 'Test'
        return name

    def log_losses(self, epoch_loss, epoch_mel_loss, epoch_stop_loss, step_number, loss_type: LossType):
        name = self.get_log_name(loss_type)
        self.writer.add_scalar(f'{name}/Total_Loss', epoch_loss, step_number)
        self.writer.add_scalar(f'{name}/Mel_Loss', epoch_mel_loss, step_number)
        self.writer.add_scalar(f'{name}/Stop_Loss', epoch_stop_loss, step_number)
        
    def log_images(self, sample_mel, sample_alignment, step_number, loss_type: LossType):
        name = self.get_log_name(loss_type)
        self.writer.add_image(f'{name}/Melspec', spectrogram_to_numpy(sample_mel.data.cpu().numpy().T), step_number)
        self.writer.add_image(f'{name}/Alignment', alignment_to_numpy(sample_alignment.data.cpu().numpy().T), step_number)

    def log_sound(self, sample_mel, input_seq, step_number, loss_type: LossType):
        name = self.get_log_name(loss_type)
        txt = seq_to_text(input_seq)
        self.writer.add_audio(f'{name}/audio', self.sc.inverse_mel_spec_to_wav(sample_mel.permute(1,0).cpu()), step_number)
        self.writer.add_text(f'{name}/text', txt, step_number)

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
