import torch
import torchaudio
import torch.nn.functional as F
import os

from torch.utils.data import Dataset

class MSCDataset(Dataset):
    """
    INPUT:
    - data_path : str -> path containing audio files in .wav format
    - classes : list[str] -> list containing ordered class labels

    """
    def __init__(self, data_path: str, classes: list[str],transform: torch.nn.Module):
        self.classes = classes
        self.transform = transform
        self.encoded_labels = self.encode_labels_(classes)
        self.data = self.unpack_audios_(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        waveform, sampling_rate, label = self.data[idx]
        if self.transform is not None:
            waveform = self.transform(waveform)
            
        return {
            'x': waveform,
            'sampling_rate': sampling_rate,
            'label': label
        }

    def unpack_audios_(self, path, target_len: int = 16000):
        audios = []
        skipped = 0

        for file_name in sorted(os.listdir(path)):
            if not file_name.endswith(".wav"):
                continue

            label = self.label_to_int(file_name.split("_")[0])
            if label is None:
                skipped += 1
                continue

            file_path = os.path.join(path, file_name)
            waveform, sr = torchaudio.load(file_path)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                sr = 16000

            length = waveform.shape[1]
            if length < target_len:
                waveform = F.pad(waveform, (0, target_len - length))
            elif length > target_len:
                waveform = waveform[:, :target_len]

            audios.append((waveform, sr, label))

        print(f"[{os.path.basename(path)}] loaded={len(audios)} skipped={skipped}")
        
        return audios

    def encode_labels_(self,labels):
        encoded = {}
        for idx,label in enumerate(labels):
            encoded[label] = idx

        return encoded

    def label_to_int(self,label):
        if label in self.encoded_labels.keys():
            return self.encoded_labels[label]
        else:
          return None