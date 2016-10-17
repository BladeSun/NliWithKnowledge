__author__ = 'Suneburg'
import numpy
import cPickle as pkl

import sys

def main(dict_path, emb_path, dim):
    with open(dict_path, 'r') as f:
       word_dict = pkl.load(f)
    num = len(word_dict)
    emb = numpy.random.randn(num, dim).astype('float32')
    emb[0, :] = numpy.random.uniform(-1, 1, dim) # for eos
    emb[1, :] = numpy.random.uniform(-1, 1, dim) # for UNK
    with open(emb_path, 'r') as f:
        for line in f:
            l = line.strip().split(' ')
            if l[0] in word_dict:
                emb[word_dict[l[0]], :] = l[1:]

    save_name = 'snli_emb_%i' % dim
    with open('%s.pkl'%save_name, 'wb') as f:
        pkl.dump(emb, f)

    print 'Done'

if __name__ == '__main__':
    emb_path = './glove.840B.300d.txt'
    dict_path = './snli_dict.pkl'
    dim = 300
    main(dict_path, emb_path, dim)




