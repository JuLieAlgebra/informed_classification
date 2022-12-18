"""Generate the data from the generative models defined in informed_classification"""
import os

import numpy as np
import pandas as pd

from informed_classification import generative_models


#### CONFIGURATION
dim = 100
samples = 10000
ratio = 0.7  # fraction data that's nominal

#### MODELS
disrupted = generative_models.DisruptedModel(dim)
nominal = generative_models.NominalModel(dim)

#### SAMPLING
disrupted_data = disrupted.sample(int(samples*(1-ratio)))
disrupted_labels = [0 for _ in range(disrupted_data.shape[0])]
nominal_data = nominal.sample(int(samples*(ratio)))
nominal_labels = [1 for _ in range(nominal_data.shape[0])]

#### PREPARING DATA TO WRITE
df = pd.DataFrame({''})

#### WRITING TO FILE
if not os.path.exists('data'):
    os.path.makedirs('data')

# maybe don't need to use the context manager
with open('data/generated_data.csv', 'w+') as f:
    f.write(df.to_csv())
