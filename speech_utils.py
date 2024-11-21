import torch
import torchaudio
from torchaudio.functional import spectrogram

# From NVIDIA TacoTron2 params
sr = 22050
n_fft = 2048
n_stft = int((n_fft//2) + 1)

frame_shift = 0.0125 # seconds
hop_length = int(n_fft/8.0)

frame_length = 0.05 # seconds  
win_length = int(n_fft/2.0)

max_mel_time = 1024

max_db = 100  
scale_db = 10
ref = 4.0
power = 2.0
norm_db = 10 
ampl_multiplier = 10.0
ampl_amin = 1e-10
db_multiplier = 1.0
ampl_ref = 1.0
ampl_power = 1.0

class SpeechConverter():
    def __init__(self, num_mels):
        self.num_mel = num_mels
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, 
            win_length=win_length,
            hop_length=hop_length,
            power=power
        )
        self.mel_scale_transform = torchaudio.transforms.MelScale(
            n_mels=self.num_mel, 
            sample_rate=sr, 
            n_stft=n_stft
        )

        self.mel_inverse_transform = torchaudio.transforms.InverseMelScale(
            n_mels=self.num_mel, 
            sample_rate=sr, 
            n_stft=n_stft
        )

        self.griffnlim_transform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        
    def pow_to_db_mel_spec(self,mel_spec):
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec,
            multiplier = ampl_multiplier, 
            amin = ampl_amin, 
            db_multiplier = db_multiplier, 
            top_db = max_db
        )
        mel_spec = mel_spec/scale_db
        return mel_spec

    def convert_to_mel_spec(self, raw_audio):
        spec = self.spec_transform(raw_audio)
        mel_spec = self.mel_scale_transform(spec)
        db_mel_spec = self.pow_to_db_mel_spec(mel_spec)
        db_mel_spec = db_mel_spec.squeeze(0)
        return db_mel_spec
    
    def inverse_mel_spec_to_wav(self, mel_spec):
        power_mel_spec = self.db_to_power_mel_spec(mel_spec)
        spectrogram = self.mel_inverse_transform(power_mel_spec)
        pseudo_wav = self.griffnlim_transform(spectrogram)
        return pseudo_wav

    def db_to_power_mel_spec(self, mel_spec):
        mel_spec = mel_spec*scale_db
        mel_spec = torchaudio.functional.DB_to_amplitude(
            mel_spec,
            ref=ampl_ref,
            power=ampl_power
            )  
        return mel_spec