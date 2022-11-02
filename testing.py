from dataclasses import dataclass
import render
import load_data
import numpy as np

data = load_data.load_synthetic(verbose=True)
dataset = load_data.PixelDataset(data['train'], verbose=True)
print(dataset[800 * 399 + 400])