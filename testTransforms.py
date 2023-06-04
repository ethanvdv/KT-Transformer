import numpy as np
from Kaleidoscope import Kaleidoscope
from phantominator import shepp_logan
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from PIL import Image, ImageOps
import csv

def kal_round(x, sigma):
    '''
    Rounds x up or down depending on whether sigma is positive or negative
    '''
    if sigma <= 0:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))

def CreatePhantom(N):
    
    #From Marlon's Code for loading Shepp_Logan
    ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
    
    ph = torch.tensor(ph.copy(), dtype=torch.float)

    ph = rearrange(ph, 'h w -> 1 1 h w')

    return ph


"""
Factors Pairs of 224:

(1, 224)
(2, 112)
(4, 56)
(7, 32)
(8, 28)
(14,16)

"""

N = 225
ph = CreatePhantom(N)

with open('/home/Student/s4532907/225-1.csv', 'r') as f:
    reader = csv.reader(f)
    instructions = list(reader)
    instructionslen = len(instructions)


dataset = []
for instr in instructions:
    row = []
    nu = int(instr[0])
    sigma = int(instr[1])
    row.append(nu)
    row.append(sigma)

    mk, check1, check2 = Kaleidoscope.KaleidoscopeShuffleIndexes(N=N, nu=nu, sigma=sigma, F=1, G=1)
    x = Kaleidoscope.ApplyKTTransform(ph, mk)

    for i in range(4*N):
        x = Kaleidoscope.ApplyKTTransform(x, mk)

        mse = torch.sum(torch.nonzero(torch.floor(x-ph)))
        err = torch.sum(torch.nonzero(torch.floor(x)))
        if err == 0:
            row.append(i+2)
            row.append("break")
            break
        if mse == 0:
            print(f"{nu},{sigma},{i+2}")
            row.append(i+2)
            break
    dataset.append(row)
    
header2 = ['nu', 'sigma', 'iterations']
with open('iterations-225.csv', 'w', encoding='UTF8', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(header2)

    writer.writerows(dataset)
        

