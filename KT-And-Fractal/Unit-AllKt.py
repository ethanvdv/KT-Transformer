import numpy as np
import matplotlib.pyplot as plt
from CreatePhantom import *
from Kaleidoscope import *

N = 176
ph = CreatePhantom(N)

'''
Method for applying the full transform:

'''
nu = 59
sigma = -1

indexes = Kaleidoscope.kaleidoscopeIndexes(N=N, nu=nu, sigma=nu)

output = Kaleidoscope.ApplyTransform(ph, indexes)

input = Kaleidoscope.pseudoInvKaleidoscope(output, indexes, 0)

'''
Method for applying the MKT:
'''

#To find shift for a specific transform:

shift = (N - sigma) / nu

print(shift)
#For some reason decimal indexes seem to work?

mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift,N=N)

mkoutput = Kaleidoscope.ApplyTransform(ph, mktindexes)

mkinput = Kaleidoscope.pseudoInvKaleidoscope(mkoutput, mktindexes, 1)

plt.figure(1)
plt.imshow(np.abs(output[0,0,:, :]))

plt.figure(2)
plt.imshow(np.abs(mkoutput[0,0,:, :]))

plt.figure(3)
plt.imshow(np.abs(indexes[0,0,:,:, :]))

plt.figure(4)
plt.imshow(np.abs(mktindexes[0,0,:,:, :]))

plt.figure(5)
plt.imshow(np.abs(input[0,0,:,:]))

plt.figure(6)
plt.imshow(np.abs(mkinput[0,0,:, :]))
plt.show()