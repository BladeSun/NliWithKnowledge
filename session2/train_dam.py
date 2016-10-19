import numpy
import os

import numpy
import os

from dam import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=100,
                     batch_size=16,
                     valid_batch_size=32,
                     validFreq=2000,
                     dispFreq=2000,
                     saveFreq=10000,
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_dam_gpu05.npz'],
        'dim_word': [300],
        'dim': [200],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.001]})
