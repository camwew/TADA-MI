# %%

import numpy as np
import joblib
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit


splits = []
train_indices_Lizard = [i for i in range(30000)]
valid_indices_Lizard = [i for i in range(30000, 35640)]
train_indices_Lizard_few = [i for i in range(8)]
valid_indices_Lizard_few = [i for i in range(8)]
train_indices_PanNuke = [i for i in range(10000)]
valid_indices_PanNuke = [i for i in range(10000, 12608)]
splits.append({'train_Lizard': train_indices_Lizard, 'valid_Lizard': valid_indices_Lizard, 'train_Lizard_few': train_indices_Lizard_few, 'valid_Lizard_few': valid_indices_Lizard_few, 'train_PanNuke': train_indices_PanNuke, 'valid_PanNuke': valid_indices_PanNuke})
joblib.dump(splits, 'splits.dat')
