import os
import librosa
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import torchvision

resize_transform = Resize((128, 128), interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT)

class CustomDataset(Dataset):
    def __init__(self, annotation_file, root_dir):
        self.root_dir = root_dir
        location_list = [d for d in os.listdir('datasets/') if os.path.isdir(os.path.join('datasets/', d)) and d != ".ipynb_checkpoints"]
        location_csv_paths = [self.root_dir + location + '/' + annotation_file for location in location_list]
        self.instances = []

        for location_csv in location_csv_paths:
            self.instances.append(pd.read_csv(location_csv))
        self.instances = pd.concat(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances.iloc[idx]
        audio_path = self.root_dir + instance['audio path'] + instance['audio file name']
        floorplan_path = self.root_dir + instance['audio path'][:-6] + 'images/' + 'image_' + instance['camera file name'][7:-5] + '.png'
        y, _ = librosa.load(audio_path, sr=44100, mono=False)
        y=y[:, :6000]
        X = librosa.stft(y, n_fft=510, win_length=64, hop_length=16)
        Xdb = librosa.amplitude_to_db(abs(X))
        Xdb = torch.from_numpy(Xdb)
        Xdb = resize_transform(Xdb)
        image = ToTensor()(Image.open(floorplan_path).convert('L'))
        image = resize_transform(image)
        image = 1 - image

        return Xdb, image