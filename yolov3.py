import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

df_train = pd.read_csv('../input/train.csv')
df_train.dropna(inplace=True)
df_train.reset_index(inplace=True, drop=True)
unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../input/unicode_translation.csv').values}

image_size = 64


class YOLO:
    def __init__(self, class_num, anchors, use_smooth_label=False, use_focal_loss=False,
                 batch_norm_decay=0.999, weight_decay=5e-4, use_static_shape=True):
        self.class_num = class_num
        self.anchors = anchors
