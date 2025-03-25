import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from braindecode.models import EEGConformer
#import python_models.Attention_FIRST_model
import importlib
import pandas as pd
from collections import OrderedDict
import math
import pickle
import os
from typing import Type,Optional
from itertools import product
import torch.nn.functional as F
import sys
import os
from concurrent.futures import ThreadPoolExecutor


sys.path.append(os.path.abspath(os.path.dirname(__file__)))