
import numpy as np

#########################

def transform(x):
    fh = 3
    fw = 3
    p1 = 1
    p2 = 1
    s = 1
    nbatch, yh, yw, _ = np.shape(x)
        
    x = np.pad(array=x, pad_width=[[0, 0], [p1,p2], [p1,p2], [0,0]], mode='constant')
    
    #########################
    
    patches = []
    for h in range(yh):
        for w in range(yw):
            patch = np.reshape(x[:, h*s:(h*s+fh), w*s:(w*s+fw), :], (nbatch, -1))
            patches.append(patch)
            
    #########################
    
    patches = np.stack(patches, axis=1)    
    pb = []
    for xb in range(params['bpa']):
        pb.append(np.bitwise_and(np.right_shift(patches.astype(int), xb), 1))
    
    patches = np.stack(pb, axis=-1)
    nbatch, npatch, nrow, nbit = np.shape(patches)
        
    #########################
    
    if (nrow % params['wl']):
        zeros = np.zeros(shape=(nbatch, npatch, params['wl'] - (nrow % params['wl']), params['bpa']))
        patches = np.concatenate((patches, zeros), axis=2)
                
    patches = np.reshape(patches, (nbatch, npatch, -1, params['wl'], params['bpa']))
                
    return patches
    
#########################

params = {
'bpa': 8,
'bpw': 8,
'adc': 8,
'skip': 1,
'cards': 0,
'stall': 0,
'wl': 128,
'bl': 128,
'offset': 128,
'sigma': 0.05,
'err_sigma': 0.,
}

cifar_out = np.load('cifar_out.npy', allow_pickle=True).item()

#########################

for layer in range(6):
    out = cifar_out[layer]
    patches = transform(out)
    nbatch, npatch, nwl, wl, xb = np.shape(patches)

    ratios = np.count_nonzero(patches, axis=(0, 1, 3, 4)) / (nbatch * npatch * wl * xb)
    print (ratios)

    avg_ratio = np.count_nonzero(patches) / np.prod(np.shape(patches))
    print (avg_ratio)

#########################

























