import os
import torch
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import video_transforms
import volume_transforms
from video_reader_frame import VideoReaderFrame  # Same as used in MAE-DFER
import torchvision.transforms as transforms

class DAiSEEMultiTaskDataset(Dataset):
    """
    Multi-Task version of VideoClsDatasetFrame for DAiSEE.
    Each video sample has four target labels:
    [engagement, boredom, confusion, frustration]
    """

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3, args=None,
                 file_ext='jpg'):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.file_ext = file_ext

        if VideoReaderFrame is None:
            raise ImportError("Decord required to read video frames.")

        df = pd.read_csv(self.anno_path)
        self.dataset_samples = df['ClipID'].tolist()
        self.label_array = df[['label_engagement', 'label_boredom', 'label_confusion', 'label_frustration']].values.astype(np.int64)

        self.data_resize = video_transforms.Compose([
            video_transforms.Resize(size=(self.short_side_size, self.short_side_size), interpolation='bilinear')
        ])
        self.data_transform = video_transforms.Compose([
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        sample = self.dataset_samples[index]
        sample_folder = sample.replace('.avi', '')  # Add this line
        full_path = os.path.join(self.data_path, sample_folder)

        buffer = self.load_video(full_path)

        if len(buffer) == 0:
            while len(buffer) == 0:
                warnings.warn(f"Video {sample} not loaded. Re-sampling...")
                index = np.random.randint(len(self.dataset_samples))
                sample = self.dataset_samples[index]
                sample_folder = sample.replace('.avi', '')  # Add this line
                full_path = os.path.join(self.data_path, sample_folder)
                buffer = self.load_video(full_path)

        buffer = self.data_resize(buffer)
        buffer = self._transform_frames(buffer)

        label = torch.tensor(self.label_array[index], dtype=torch.long)  # shape (4,)

        return buffer, label, index

    def load_video(self, folder_path):
        """Load video frames from a folder as PIL Images."""
        if not os.path.exists(folder_path) or len(os.listdir(folder_path)) == 0:
            return []

        all_frames = sorted([
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.endswith(self.file_ext)
        ])

        frame_indices = list(range(0, len(all_frames), self.frame_sample_rate))
        frame_indices = frame_indices[:self.clip_len]

        # Pad with last frame if not enough
        while len(frame_indices) < self.clip_len:
            frame_indices.append(frame_indices[-1])

        frames = []
        for idx in frame_indices:
            try:
                img = Image.open(all_frames[idx]).convert("RGB")
                frames.append(img)
            except:
                continue

        return frames

    def _transform_frames(self, frames):
        buffer = [self.args.aug_transform(frame) if self.mode == 'train' else frame for frame in frames]
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer, dim=0)  # (T, C, H, W)
        buffer = buffer.permute(1, 0, 2, 3)  # (C, T, H, W)
        buffer = self.data_transform(buffer)
        return buffer
