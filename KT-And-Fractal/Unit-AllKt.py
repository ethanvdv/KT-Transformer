import numpy as np
import matplotlib.pyplot as plt
from CreatePhantom import *
from Kaleidoscope import *

'''

How to use the Kaleidoscope Class:


Firstly you need to create the tensors of the indexes of the transform. 

This is done by calling either: 

kaleidoscopeIndexes

or

MKTkaleidoscopeIndexes

After this you need to call the "ApplyTransform" function.

For the pseudo-inverse note:

Type = 0 - Full Transform
Type = 1 - Multiplication Transform

I *believe* that the nu and shift values may be the wrong way around.

E.g. to apply a correct full 3-1 tranform, nu = 59 and sigma = 1.

However to get this with the Multiplication Transform we have to multiple the indexes
by shift = 59


TODO: 
-- Work out the best transforms for the fractal.
-- Find the equivalences
-- Find the subtle differences
'''


N = 176
ph = CreatePhantom(N)

# Transforms needed for Fractal
# [3, 5, 7, 25, 35, 59]
# [-3, -5, -7, -25, -35, -59]

'''
Method for applying the full transform:

'''
nu = 11
sigma = 1

indexes = Kaleidoscope.kaleidoscopeIndexes(N=N, nu=nu, sigma=sigma)

output = Kaleidoscope.ApplyFullTransform(ph, indexes)

input = Kaleidoscope.pseudoInvKT(output, indexes)

'''
Method for applying the MKT:

For some reason decimal shifts seem to be working? 
shift = np.ceil((N - sigma) / nu)

'''

#To find shift for a specific transform:
shift = ((N + sigma) / nu)


mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift,N=N)
print(mktindexes.shape)

mkoutput = Kaleidoscope.ApplyMKTransform(ph, mktindexes)

mkinput = Kaleidoscope.pseudoInvMKTransform(mkoutput, mktindexes)

plt.figure(1)
plt.imshow(np.abs(output[0,0,:, :]))
plt.title("Full Transform")

plt.figure(2)
plt.imshow(np.abs(mkoutput[0,0,:, :]))
plt.title("MKT")

plt.figure(3)
plt.imshow(np.abs(indexes[0,0,:,:, :]))
plt.title("Full Transform - Indexes")

plt.figure(4)
plt.imshow(np.abs(mktindexes[0,0,:,:, :]))
plt.title("MKT - Indexes")

plt.figure(5)
plt.imshow(np.abs(input[0,0,:,:]))
plt.title("Full Transform - Pseudo Inverse")


plt.figure(6)
plt.imshow(np.abs(mkinput[0,0,:, :]))
plt.title("MKT - Pseudo Inverse")

plt.show()