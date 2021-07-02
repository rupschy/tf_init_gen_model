import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


ANNOTATIONS_FILE = "/Users/johnrupsch/Models/tf_init_model/esc50.csv"
SPECTROGRAMS_DIR = "/Users/johnrupsch/Models/tf_init_model/Spectrograms/"

if __name__ == "__main__":
    df = pd.read_csv(SPECTROGRAMS_DIR)
    X = df.drop("amplitude", axis = 1).values