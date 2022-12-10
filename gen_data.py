"""Generate the data from the generative models defined in informed_classification"""
import numpy as np
from informed_classification import generative_models

dim = 100
samples = 10000
ratio = 0.7  # fraction data that's nominal
disrupted = generative_models.DisruptedModel(dim)
nominal = generative_models.NominalModel(dim)

disrupted_data = disrupted.sample(int(samples*(1-ratio)))
disrupted_labels = [0 for _ in range(disrupted_data.shape[0])]
nominal_data = nominal.sample(int(samples*(ratio)))
nominal_labels = [1 for _ in range(nominal_data.shape[0])]

import pandas as pd

df = pd.DataFrame()