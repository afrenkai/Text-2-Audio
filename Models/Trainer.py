from torch import nn
import torch


# Use this to check output/input shapes for the model
def print_shape_hook(module: nn.Module, input: torch.Tensor, output:torch.Tensor|tuple):
    if isinstance(output, tuple):
        print(module._get_name())
    else:
        print(f'{module._get_name()}, {output.shape}')

class Trainer():
    def __init__(self, model : nn.Module, epochs, optimizer, 
                 criterion,  train_dl, val_dl, device, 
                 checkpoint_name, log_shape=False):
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

        # # does a test pass and shows the shapes for the input and output for each of the layers
        # if log_shape:
        #     _tmp_handlers = []
        #     self.model.eval()
        #     for mod in self.model.modules():
        #         if isinstance(mod, nn.Module):
        #             handle = mod.register_forward_hook(print_shape_hook)
        #             _tmp_handlers.append(handle)
        #     padded_text_seqs, _, padded_mel_specs, _ = next(iter(train_dl))
        #     with torch.no_grad():
        #         self.model.forward(padded_text_seqs, padded_mel_specs, 0)
        #     for handle in _tmp_handlers:
        #         handle.remove()


    def train(self):
        print("Starting training")
        # trains self.model with certain params
        for epoch in range(self.max_epochs):
            self.model.train()
            running_loss = 0.0
            for padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens in self.train_dl:
                padded_text_seqs, padded_mel_specs = padded_text_seqs.to(self.device), padded_mel_specs.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(padded_text_seqs, padded_mel_specs, 1)
                loss = self.criterion(outputs, padded_mel_specs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(self.train_dl)
            # Validation
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens in self.val_dl:
                    padded_text_seqs, padded_mel_specs = padded_text_seqs.to(self.device), padded_mel_specs.to(self.device)
                    outputs = self.model(padded_text_seqs, padded_mel_specs, 0)
                    loss = self.criterion(outputs, padded_mel_specs)
                    running_val_loss += loss.item()
            epoch_val_loss = running_val_loss / len(self.val_dl)

            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                self.best_model_state_dict = self.model.state_dict()
                torch.save(self.best_model_state_dict, self.cp_name)
            print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {epoch_loss}, Val Loss: {epoch_val_loss}")




