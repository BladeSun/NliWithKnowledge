__author__ = 'Suneburg'
import numpy
import cPickle as pkl

import sys

def main(dict_path, emb_path, dim):
    with open(dict_path, 'r') as f:
       dict = pkl.load(f)
    num = len(dict)
    emb = numpy.array(num, dim).astype('float32')
    emb[0, :] = numpy.random.uniform(-1, 1, dim) # for eos
    emb[1, :] = numpy.random.uniform(-1, 1, dim) # for UNK
    with open(emb_path, 'r') as f:
        for line in f:
            l = line.strip().split(' ')
            if l[0] in dict:
                emb[dict[l[0]], :] = l[1:]

    save_name = 'snli_emb_' % dim
    with open('%s.pkl'%save_name, 'wb') as f:
        pkl.dump(emb, f)

    print 'Done'

if __name__ == '__main__':
    dict_path = ''
    emb_path = ''
    dim = 300
    main()




