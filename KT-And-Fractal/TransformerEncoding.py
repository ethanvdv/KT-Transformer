import numpy as np
import matplotlib.pyplot as plt
from CreatePhantom import *
from Kaleidoscope import *
from einops import rearrange

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
Method for applying the MKT:

For some reason decimal shifts seem to be working? 
shift = np.ceil((N - sigma) / nu)

'''

# for i in [25, 35, 59]:
sigma = 1
mkinput = ph
# #To find shift for a specific transform:
# shift = np.ceil((N - sigma) / nu)
for nu in [5, 7, 25, 35, 59]:
    i = nu
    if np.mod(N - sigma,nu) == 0:
        case = 1
        shift = (N - sigma)/nu
        print(nu)
        print(shift)

    else:
        case = 2
        shift = (N + sigma)/nu
        print(nu)
        print(shift)

    mktindexes = Kaleidoscope.MKTkaleidoscopeIndexes(shift=shift,N=N)

    mkoutput = Kaleidoscope.ApplyMKTransform(mkinput, mktindexes)


    '''
    Patchify and un-Patchify
    '''

    # print(torch.cat((mkoutput, mkoutput[0, 0, -1, -1]),2).shape)

    patch = nu

    if case == 1:
        

        # Case 1: Need to remove a column and row

        newtensor = rearrange(mkoutput[0, :, :-1, :-1], 'c (h k1) (w k2) -> (h w) c k1 k2', k1=patch, k2=patch)
        print(newtensor.shape)
        undotensor = rearrange(newtensor, '(h w) c k1 k2-> c (h k1) (w k2)', h= N // patch, k1=patch, k2=patch)

        mkoutput[0, :, :-1, :-1] = undotensor


    #Case 2: We need to add one column and row
    else:

        #Build Extra rows    
        newtensor = torch.zeros(1, 177, 177)
        newtensor[:,:-1,:-1] = mkoutput[0,:,:,:]
        newtensor[:,:-1,-2:-1] = mkoutput[0, 0, :, -2:-1]
        newtensor[:,-2:-1,:-1] = mkoutput[0, 0, -2:-1, :]
        

        newtensor = rearrange(newtensor, 'c (h k1) (w k2) -> (h w) c k1 k2', k1=patch, k2=patch)
        print(newtensor.shape)        

        undotensor = rearrange(newtensor, '(h w) c k1 k2-> c (h k1) (w k2)', h= (N + 1) // patch, k1=patch, k2=patch)

        mkoutput[0, :, :, :] = undotensor[:, :-1, :-1]


    mkinput = Kaleidoscope.pseudoInvMKTransform(mkoutput, mktindexes)

plt.figure(1)
plt.imshow(np.abs(mkinput[0,0,:, :]))
plt.title("MKT")
plt.show()


#Loss free
print(torch.sum(torch.nonzero(ph-mkinput)))
