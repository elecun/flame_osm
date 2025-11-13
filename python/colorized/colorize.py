import torch, functools
import fastai.basic_train
torch.serialization.add_safe_globals([
    functools.partial,
    fastai.basic_train.Recorder
])
torch.serialization.add_safe_globals([functools.partial])

import sys
import os

from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import get_image_colorizer

# # GPU 설정 (CUDA 사용)
from enum import Enum
class DeviceId(Enum):
    CPU = 'cpu'
    CUDA = 'cuda'
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0 사용

device.set(device=DeviceId.CUDA)
colorizer = get_image_colorizer(artistic=False)
colorizer.plot_transformed_image('sample.jpg', render_factor=35)

