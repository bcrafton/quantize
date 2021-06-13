
import numpy as np
import tensorflow as tf
from layers import *
from load import Loader
import time
import pandas as pd

####################################

gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[1]
tf.config.experimental.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)

####################################

def ld_to_dl(ld):
    dl = {}

    for i, d in enumerate(ld):
        for key in d.keys():
            value = d[key]
            if i == 0:
                dl[key] = [value]
            else:
                dl[key].append(value)

    return dl

###########

results = np.load('results_lrs.npy', allow_pickle=True)

# results1 = np.load('imagenet_results_extra.npy', allow_pickle=True)
# results2 = np.load('imagenet_results.npy', allow_pickle=True)
# results = results1.tolist() + results2.tolist()

results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

###########

query = '(example == 0)' 
samples = df.query(query)

lrss = np.array(samples['lrs'])
hrss = np.array(samples['hrs'])
ratios = np.array(samples['ratio'])
layers = np.array(samples['layer_id'])
errors = np.array(samples['error'])
cim_errors = np.array(samples['cim_error'])
cim_means  = np.array(samples['cim_mean'])
cim_stds = np.array(samples['cim_std'])
cards  = np.array(samples['cards'])
skips  = np.array(samples['skip'])
rpr_alloc = samples['rpr_alloc'].tolist()
threshs = np.array(samples['thresh'])

N = len(lrss)
print (N)
#print (sigmas[1000])
#assert (False)

###########

stats = {}
for i in range(N):
    card  = cards[i]
    skip  = skips[i]
    rpr   = rpr_alloc[i]
    thresh = threshs[i]
    hrs = hrss[i]
    lrs = lrss[i]
    ratio = ratios[i]
    id    = layers[i]
    error = errors[i]
    cim_error = cim_errors[i]
    cim_mean = cim_means[i]
    cim_std = cim_stds[i]

    key = (card, hrs, lrs, ratio, thresh)
    if key not in stats.keys():
        stats[key] = {}
    stats[key][id] = {'error': error, 'cim_error': cim_error, 'cim_mean': cim_mean, 'cim_std': cim_std}

####################################

for key in stats:
    print (stats[key])

####################################

def imagenet_predict(error):

    ####################################

    def quantize_np(x, low, high):
        scale = np.max(np.absolute(x)) / high
        x = x / scale
        x = np.floor(x)
        x = np.clip(x, low, high)
        return x

    ####################################

    def predict(model, x, y):
        pred_logits = model.predict(x)
        pred_label = tf.argmax(pred_logits, axis=1)
        correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
        return correct

    ####################################

    def run_val(model):

        total = 50000
        total_correct = 0
        total_loss = 0
        batch_size = 50

        load = Loader('/home/bcrafton3/Data_HDD/keras_imagenet/keras_imagenet_val/', total // batch_size, batch_size, 8)
        start = time.time()

        for batch in range(0, total, batch_size):
            while load.empty(): pass # print ('empty')
            
            x, y = load.pop()
            correct = predict(model, x, y)

            total_correct += correct.numpy()
            acc = round(total_correct / (batch + batch_size), 3)
            avg_loss = total_loss / (batch + batch_size)
            
            if (batch + batch_size) % (batch_size * 100) == 0:
                img_per_sec = (batch + batch_size) / (time.time() - start)
                print (batch + batch_size, img_per_sec, acc, avg_loss)

        load.join()
        acc = round(total_correct / total, 3)
        return acc

    ####################################

    weights = np.load('trained_weights.npy', allow_pickle=True).item()

    ####################################

    m = model(layers=[
    conv_block((7,7,3,64), 2, weights=weights, train=False, error=error),

    max_pool(2, 3),

    res_block1(64,   64, 1, weights=weights, train=False, error=error),
    res_block1(64,   64, 1, weights=weights, train=False, error=error),

    res_block2(64,   128, 2, weights=weights, train=False, error=error),
    res_block1(128,  128, 1, weights=weights, train=False, error=error),

    res_block2(128,  256, 2, weights=weights, train=False, error=error),
    res_block1(256,  256, 1, weights=weights, train=False, error=error),

    res_block2(256,  512, 2, weights=weights, train=False, error=error),
    res_block1(512,  512, 1, weights=weights, train=False, error=error),

    avg_pool(7, 7),
    dense_block(512, 1000, weights=weights, train=False, error=error)
    ])
    layer.layer_id = 0
    layer.weight_id = 0

    return run_val(m)

####################################

for i, key in enumerate(sorted(stats.keys())):
    acc = imagenet_predict(stats[key])
    stats[key]['acc'] = acc
    print ('%d/%d' % (i, len(stats.keys())), key, stats[key]['acc'])
    
np.save('acc', stats)

####################################





