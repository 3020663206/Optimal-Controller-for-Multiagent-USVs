import model_def
import torch
def num():
    a = 1
    b = 2
    c = 2
    return a,b,c

r = num()
print(r[1])
print(torch.__version__)
print(dir(torch.distributions))