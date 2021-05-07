import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
import sys

def load_se_module(se_train_yaml):
    # Load hyperparameters file with command-line overrides
    with open(se_train_yaml) as fin:
        hparams = load_hyperpyyaml(fin)

    # initialize a new model
    model = hparams["model"]
    model.eval()

    # put model on specified device
    device = hparams['run_opts']['device']
    model.to(device)

    # Load model weights
    ckpt_finder = hparams["checkpointer"]
    best_ckpt = ckpt_finder.find_checkpoint(max_key="stoi").paramfiles["model"]
    with torch.no_grad():
        sb.utils.checkpoints.torch_parameter_transfer(model, best_ckpt, device)

    def compute_feats(wavs):
        # Log-spectral features
        feats = hparams['compute_STFT'](wavs)
        feats = sb.processing.features.spectral_magnitude(feats, power=0.5)

        # Log1p reduces the emphasis on small differences
        feats = torch.log1p(feats)

        return feats

    def enhance_audio(noise_wav):
        noise_wav = noise_wav.to(device)
        noisy_feats = compute_feats(noise_wav)

        # Masking is done here with the "signal approximation (SA)" algorithm.
        # The masked input is compared directly with clean speech targets.
        with torch.no_grad():
            mask = model(noisy_feats)
            predict_spec = torch.mul(mask, noisy_feats)

        # Also return predicted wav, for evaluation. Note that this could
        # also be used for a time-domain loss term.
        predict_wav = hparams['resynth'](
            torch.expm1(predict_spec), noise_wav
        )

        # Return a dictionary so we don't have to remember the order
        return predict_wav

    return enhance_audio