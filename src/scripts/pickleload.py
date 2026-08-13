import pickle
import torch

with open("/home/jonas/Documents/schnetpack_fork/tests/testdata/newton_step_golden.pt", "rb") as f:
    my_object = pickle.load(f)

torch.load("/home/jonas/Documents/schnetpack_fork/tests/testdata/newton_step_golden.pt")