import os
import random
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add fairseq root for HuBERT
import sys
fairseq_root = os.path.abspath("fairseq")
if fairseq_root not in sys.path:
    sys.path.insert(0, fairseq_root)
from fairseq import checkpoint_utils

# FAISS utilities
from faiss_index_creation import load_index, get_centroids_index, apply_centroids_to_audios

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
torch.cuda.empty_cache()

HUBERT_CHECKPOINT = 'pretrained_models_anon_xv/mhubert/checkpoint_best.pt'
KM_MODEL_INDEX = 'pretrained_models_anon_xv/mhubert/mhubert147_faiss.index'
TRAIN_LIST = 'scp/librispeech_100/librispeech_100_wav16k_norm_train.lst'
DEV_LIST   = 'scp/librispeech_100/librispeech_100_wav16k_norm_dev.lst'
KM_CLASSES = 1000
SSL_DIM    = 768
BATCH_SIZE = 8
LR         = 2e-5
MAX_EPOCHS = 1000
PATIENCE   = 20
LOG_DIR    = 'runs/mhubert_train'
CKPT_DIR   = 'checkpoints/mhubert_train'
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_hubert(checkpoint_path: str):
    """
    Load HuBERT model and task from Fairseq checkpoint.
    """
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    hubert = models[0]
    hubert.remove_pretraining_modules()
    hubert.to(DEVICE).eval()
    return hubert, task


def length_to_mask(lengths: torch.LongTensor, max_len: int = None):
    """
    Create binary mask for lengths tensor.
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    idx = torch.arange(max_len, device=lengths.device)
    mask = idx.unsqueeze(0).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------
class SoftPredictor(nn.Module):
    """
    Project HuBERT features into k-means clusters.
    """
    def __init__(self, hubert: nn.Module, km_classes: int, ssl_dim: int):
        super().__init__()
        self.ssl_model = hubert
        self.W = nn.Parameter(torch.randn(km_classes, ssl_dim))

    def forward(self, wav_batch: torch.Tensor):
        """
        Args:
            wav_batch: [batch, 1, time]
        Returns:
            logits_scaled: [batch, frames, km_classes]
        """
        wav = wav_batch.squeeze(1)
        out = self.ssl_model(wav, mask=False, features_only=True)
        x = out['x']  # [batch, frames, ssl_dim]
        x_norm = F.normalize(x, dim=-1)
        W_norm = F.normalize(self.W, dim=-1)
        logits = F.linear(x_norm, W_norm)
        return logits * 10.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class FineTuneDataset(Dataset):
    def __init__(self, wavlist, max_length=20.0):
        """
        Args:
            wavlist (str): Path to the list of WAV file paths.
            max_length (float): Maximum audio length in seconds.
        """
        # Read all file paths from the list
        self.wavnames = [line.strip() for line in open(wavlist, 'r')]
        self.max_length = max_length

    def __getitem__(self, idx):
        wavpath = self.wavnames[idx]
        try:
            wav, sample_rate = torchaudio.load(wavpath)
            max_samples = int(self.max_length * sample_rate)

            # If the audio is longer than allowed, randomly crop a segment of max_length duration.
            if wav.shape[1] > max_samples:
                start = random.randint(0, wav.shape[1] - max_samples)
                wav = wav[:, start:start + max_samples]

            return wav, wavpath.split('/')[-1].split('.')[0]
        except Exception as e:
            # In case of error, return None so that collate_fn can filter it out.
            return None

    def __len__(self):
        return len(self.wavnames)

    def collate_fn(self, batch):
        # Filter out None entries (in case any __getitem__ call returned None)
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            raise ValueError("All batch items are None.")

        # Unpack the batch into wave tensors and identifiers
        wavs, wavnames = zip(*batch)
        # Pad all waveforms in the batch to the length of the longest one
        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]
        output_wavs = [F.pad(wav, (0, max_len - wav.shape[1]), 'constant', 0) for wav in wavs]

        return torch.stack(output_wavs), wavnames, [wav.shape[1] / max_len for wav in wavs]

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Load HuBERT and feature-extractor for pseudo-labels
    hubert, task = load_hubert(HUBERT_CHECKPOINT)
    feat_models, _, _ = checkpoint_utils.load_model_ensemble_and_task([HUBERT_CHECKPOINT])
    feat_model = feat_models[0].eval().to(DEVICE)

    # Load FAISS index
    faiss_index, faiss_ivf = load_index(KM_MODEL_INDEX)

    # Model, optimizer, loss
    model = SoftPredictor(hubert, KM_CLASSES, SSL_DIM).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Data loaders
    train_ds = FineTuneDataset(TRAIN_LIST)
    dev_ds   = FineTuneDataset(DEV_LIST)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=train_ds.collate_fn, num_workers=4)
    dev_loader   = DataLoader(dev_ds,   BATCH_SIZE, shuffle=False,
                              collate_fn=dev_ds.collate_fn,   num_workers=4)

    # TensorBoard logger
    writer = SummaryWriter(LOG_DIR)

    best_val = float('inf')
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS+1):
        # Training
        model.train()
        total_train = 0.0
        for wavs, _, lengths in train_loader:
            wavs = wavs.to(DEVICE)
            # Pseudo-labels via FAISS
            with torch.no_grad():
                feats = feat_model.extract_features(source=wavs.squeeze(1), mask=False, output_layer=6)[0]
                # Frame counts per sample
                frame_lens = [f.shape[0] for f in feats]
                flat_feats = feats.reshape(-1, feats.shape[-1]).cpu().numpy()
                _, centroids = get_centroids_index(flat_feats, faiss_index, faiss_ivf)
                vecs = apply_centroids_to_audios(frame_lens, centroids)
                km_labels = torch.stack([torch.LongTensor(v).to(DEVICE) for v in vecs])

            # Forward
            logits = model(wavs)  # [batch, frames, km_classes]
            frames = logits.size(1)
            lengths_tensor = torch.tensor(frame_lens, device=DEVICE)
            mask = length_to_mask(lengths_tensor, max_len=frames).unsqueeze(-1)

            loss = criterion((logits * mask).transpose(2, 1), (km_labels * mask.squeeze(-1)).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train, epoch)

        # Validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for wavs, _, lengths in dev_loader:
                wavs = wavs.to(DEVICE)
                feats = feat_model.extract_features(source=wavs.squeeze(1), mask=False, output_layer=6)[0]
                frame_lens = [f.shape[0] for f in feats]
                flat_feats = feats.reshape(-1, feats.shape[-1]).cpu().numpy()
                _, centroids = get_centroids_index(flat_feats, faiss_index, faiss_ivf)
                vecs = apply_centroids_to_audios(frame_lens, centroids)
                km_labels = torch.stack([torch.LongTensor(v).to(DEVICE) for v in vecs])

                logits = model(wavs)
                frames = logits.size(1)
                lengths_tensor = torch.tensor(frame_lens, device=DEVICE)
                mask = length_to_mask(lengths_tensor, max_len=frames).unsqueeze(-1)

                total_val += criterion((logits * mask).transpose(2,1), (km_labels * mask.squeeze(-1)).long()).item()

        avg_val = total_val / len(dev_loader)
        writer.add_scalar('Loss/Val', avg_val, epoch)
        print(f"Epoch {epoch:03d} | Train {avg_train:.4f} | Val {avg_val:.4f}")

        # Checkpoint
        ckpt = os.path.join(CKPT_DIR, f"epoch{epoch:03d}_val{avg_val:.4f}.pt")
        torch.save(model.state_dict(), ckpt)

        # Early stopping
        if avg_val < best_val:
            best_val = avg_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs. Stopping.")
                break

    writer.close()

if __name__ == '__main__':
    main()
