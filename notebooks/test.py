import sys
sys.path.insert(0,'..')

from datasets import load_dataset

import pandas as pd
import numpy as np
from PIL import Image
from emsim.geant.dataset import GeantElectronDataset
from transformers import MaskFormerImageProcessor

pixelated_file = '/home/zero/emsim/segmentation/pixelated_1pt25um_tracks_thinned_4um_back_20k_300keV.txt'
true_pixelated_file = '/home/zero/emsim/segmentation/true_pixelated_1pt25um_thinned_4um_back_20k_300keV.txt'

from transformers import MaskFormerImageProcessor

processor = MaskFormerImageProcessor(reduce_labels=True, ignore_index=255)

import albumentations as A

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

train_transform = A.Compose([
    # A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])


events_per_image_range = (12, 16)
dataset = GeantElectronDataset(pixels_file=pixelated_file, undiffused_file=true_pixelated_file, events_per_image_range=events_per_image_range, processor=processor, transform=train_transform)


from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset, batch_size=None)

from transformers import MaskFormerForInstanceSegmentation
from itertools import islice
from PIL import Image

current_chunk = islice(train_dataloader, 1)
# batch = next(current_chunk)

print(next(current_chunk))
