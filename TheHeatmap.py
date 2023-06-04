import numpy as np
from Kaleidoscope import Kaleidoscope
from phantominator import shepp_logan
import matplotlib.pyplot as plt
import torch
from einops import rearrange, repeat
from torchmetrics import PeakSignalNoiseRatio
import csv

def CreatePhantom(N):
    
    #From Marlon's Code for loading Shepp_Logan
    # ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
    ph = np.random.randint(0,255, size=(N,N))
    plt.imsave("originalimage2.png", ph, cmap='inferno', vmin=0, vmax=np.max(ph))
    ph = torch.tensor(ph.copy(), dtype=torch.float)

    ph = rearrange(ph, 'h w -> 1 1 h w')

    return ph

def gcd(p,q):
# Create the gcd of two positive integers.
    while q != 0:
        p, q = q, p%q
    return p

def is_coprime(x, y):
    return gcd(x, y) == 1
               
def test2(N,nu, sigma):
    if (np.mod(N,nu) == 0) and (is_coprime(np.abs(N/nu),np.abs(sigma)) == True):
        if nu > 0:
            return True
        if (nu < 0) and (sigma % nu == 0):
            return True
        else:
            return False
    else:
        return False
 

def test1(N, nu,sigma):
    if (nu > 0):
        if (np.mod(N + sigma,nu) == 0) :
            if (nu > np.abs(sigma)):
                if is_coprime(N, (N+sigma)/nu):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    if (nu < 0):
        if (sigma < 0):
            L = N//nu
            if (L*nu - sigma) % N == 0:
                if gcd(L, N) == 1:
                    return True
                else:
                    return False
            else: 
                return False
        if (sigma > 0):
            L = np.ceil(N/nu)
            if (L*nu - sigma) % N == 0:
                if gcd(L, N) == 1:
                    return True
                else:
                    return False
            else: 
                return False
        
        else:
            return False      
    else:
        return False


torch.cuda.set_device('cuda:0')


N = 225
forwardarray = np.zeros([2*N,2*N])
backwardarray = np.zeros([2*N,2*N])

good = np.zeros([2*N,2*N])
good2 = np.zeros([2*N,2*N])
good3 = np.zeros([2*N,2*N])
good4 = np.zeros([2*N,2*N])
good5 = np.zeros([2*N,2*N])

ph = CreatePhantom(N).to('cuda')

nus = np.arange(-N,N)
nus = nus[nus != 0]
sigmas = nus[nus != 0]

count = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
test1count = 0
test2count = 0
test3count = 0
badTest1count = 0
badTest2count = 0
badTest3count = 0
dataset = []
dataset2 = []

for nu in nus:
    for sigma in sigmas:
        row = []
        count += 1
        mktindexes, check1 = Kaleidoscope.KalIndexes(N=N, nu=nu, sigma=sigma)

        #find forward error
        x = Kaleidoscope.ApplyKTTransform(ph, mktindexes)
        ph2 = Kaleidoscope.pseudoInvKTransform(x, mktindexes)

        mse = torch.sum(torch.nonzero(torch.floor(ph-ph2)))
        # forwardarray[nu+N,sigma+N] = mse
        
            
        #backward transform error
        x = Kaleidoscope.pseudoInvKTransform(ph, mktindexes)
        ph2 = Kaleidoscope.ApplyKTTransform(x, mktindexes)
        mse1 = torch.sum(torch.nonzero(torch.floor(ph-ph2)))
                    
        # if check1 == N:
        #     good4[nu+N,sigma+N] = 255
            

        if (mse1 == 0) and (mse == 0):
            good3[nu+N,sigma+N] = 255
            row.append(nu)
            row.append(sigma)
            
            count3 += 1
            
            t1 = test1(N,nu,sigma)
            t2 = test2(N,nu,sigma)
            # t4 = test4(N,nu,sigma)
            if t1:
                good3[nu+N,sigma+N] = 100
                test1count +=1
            elif t2:
                good3[nu+N,sigma+N] = 150
                test2count += 1
            row.append(t1)
            row.append(t2)
            if t1 or t2:
                count1 += 1
            else:
                row.append("Missing")
                count5 += 1
            # row.append(t1 + t2 + t3)
            dataset.append(row)
        else: 
            t1 = test1(N,nu,sigma)
            if t1:
                badTest1count += 1
            t2 = test2(N,nu,sigma)
            if t2:
                badTest2count +=1
            if t1 or t2:
                count2 += 1
                row.append(nu)
                row.append(sigma)
                row.append(test1(N,nu,sigma))
                row.append(test2(N,nu,sigma))
                row.append(check1)
                row.append(mse)
                row.append(mse1)
                dataset2.append(row)
            
        
        if bool(check1 == N) ^ bool((mse1 == 0) and (mse == 0)):
            count4 += 1
            print(f"nu - sigma, {(nu,sigma)}")
            good5[nu+N,sigma+N] = 255

    
# plt.imsave("out1/forward-heatmap.png", forwardarray/np.max(forwardarray), cmap='inferno')
# plt.imsave("out1/forward-good.png", good, cmap='inferno', vmin=0, vmax=np.max(good))

# plt.imsave("out1/backwawrd-heatmap.png", backwardarray/np.max(backwardarray), cmap='inferno')
# plt.imsave("out1/backward-good.png", good2, cmap='inferno', vmin=0, vmax=np.max(good2))

# plt.imsave("out1/check-good.png", good4, cmap='inferno', vmin=0, vmax=np.max(good4))

plt.imsave("out1/good.png", good3, cmap='inferno', vmin=0, vmax=np.max(good3))
# plt.imsave("out1/good2.png", good3, cmap='gray', vmin=0, vmax=np.max(good3))
# plt.imsave("out1/good3.png", good3)

plt.imsave("out1/difference.png", good5, cmap='inferno', vmin=0, vmax=np.max(good5))


header = ['nu', 'sigma', 'test1', 'test2','test3']
with open(f'{N}.csv', 'w', encoding='UTF8', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(header)

    writer.writerows(dataset)
    
     
        

header = ['nu', 'sigma', 'test1', 'test2','test3']
with open(f'{N}-2.csv', 'w', encoding='UTF8', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(header)

    writer.writerows(dataset2)
    


print(f"Count 1 {count1} - {100* count1/count}% - Test 1 or Test 2 Passed")
print(f"Count 2 {count2} - {100* count2/count}% - Not Invertible using Direct Test, but test 1 or 2 passes")
print(f"Count 3 {count3} - {100* count3/count}% - Invertible by Direct Test")
print(f"Count 4 {count4} - {100* count4/count}% - If value is not covered by index check xor direct test")
print(f"Count 5 {count5} - {100* count5/count}% - Missing Transforms")

print(f"Test 1 Count {test1count} - bad {badTest1count}")
print(f"Test 2 Count {test2count} - bad {badTest2count}")
print(f"Test 3 Count {test3count} - bad {badTest3count}")
print(f"Total Count {count}")    

print("Done")

