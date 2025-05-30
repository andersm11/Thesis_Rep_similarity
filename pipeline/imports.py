import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from shallow_fbcsp import ShallowFBCSPNet
from braindecode.models import EEGConformer
import Attention_models
import importlib
import pandas as pd
from collections import OrderedDict
import math
import pickle
import os
from typing import Type,Optional
from itertools import product
import collapsed_shallow_fbscp
import torch.nn.functional as F