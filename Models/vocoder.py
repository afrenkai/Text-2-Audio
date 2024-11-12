import torch
import torchaudio
from TtsTransformers import TTSTransformers
from TtsSimpleTest import TTS_Simple
import TTS_DataLoader
from Trainer import TTS_Loss
from torchaudio.transforms import MelSpectrogram
import argparse
import os


class Vocoder:
    def __init__(self, model_class, model_config, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model_class(**model_config).to(self.device)

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        self.mel_transform = MelSpectrogram(
            sample_rate=22050, n_mels=model_config['mel_bins'], hop_length=256)


    def generate_audio(self, text):
        text_seq = TTS_DataLoader.text_to_seq_char_level(text).to(self.device)

        with torch.no_grad():
            mel_spec, _ = self.model.inference(text_seq.unsqueeze(0))

        waveform = self.mel_spec_to_waveform(mel_spec.squeeze(0))

        return waveform

    def mel_spec_to_waveform(self, mel_spec):
        inverse_transform = torchaudio.transforms.GriffinLim(n_iter=32)
        waveform = inverse_transform(mel_spec)
        return waveform

    def save_audio(self, waveform, output_path):
        torchaudio.save(output_path, waveform.unsqueeze(0), 22050)


def main():
    parser = argparse.ArgumentParser(description="Vocoder Inference")
    parser.add_argument('--text', type=str, required=True, help="Input text for TTS")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--output', type=str, default="output.wav", help="Output audio file name")
    parser.add_argument('--model', type=str, choices=['simple', 'transformer'], default='transformer',
                        help="Model type to use (simple or transformer)")

    args = parser.parse_args()

    common_config = {
        'vocab_size': TTS_DataLoader.symbols_len,
        'mel_bins': 128,
        'embedding_dim': 64,
    }

    transformer_config = {
        **common_config,
        'n_heads_enc': 8,
        'n_heads_dec': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_ffn': 512
    }

    simple_config = {
        **common_config,
        'enc_out_size': 128,
        'dec_lstm_out_size': 256
    }

    # Model selection
    model_classes = {
        'simple': TTS_Simple,
        'transformer': TTSTransformers
    }

    model_class = model_classes[args.model]
    model_config = transformer_config if args.model == 'transformer' else simple_config

    vocoder = Vocoder(model_class, model_config, args.checkpoint)

    waveform = vocoder.generate_audio(args.text)
    vocoder.save_audio(waveform, args.output)

    print(f"Audio generated and saved to {args.output}")


if __name__ == "__main__":
    main()
