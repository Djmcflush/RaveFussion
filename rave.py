import librosa as li
import riffusion
import cached_conv as cc
import soundfile as sf


class Raven:
    """This is a Rave based Audio to Audio autoencoder"""

    def __init__(self):
        gin = riffusion.gin
        gin.parse_config_file("configs/v2.gin")
        pretrained = riffusion.RAVE.load_from_checkpoint("best.ckpt")
        self.rave_model = pretrained.eval()

    def load_audio(self, wav_file, output_path, lantent_embedding_bias=None):
        """Load an Audio file and Do a decoding run"""
        wav_form = li.load(wav_file)[0]
        x = torch.from_numpy(wav_form).reshape(1, 1, -1)
        z = self.rave_model.encode(x)
        if not lantent_embedding_bias:
            bias = np.random.randint(0, z.shape[1])
            print(f"random bias lantenet variable selected {bias}")
        z[:, bias] += torch.linspace(-2, 2, z.shape[-1])
        y = self.rave_model.decode(z).detach().numpy().reshape(-1)
        sf.write(output_path, y, 44100)
        print(f"Success new Audio File @ {output_path}")
