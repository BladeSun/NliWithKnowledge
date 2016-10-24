__author__ = 'Suneburg'
import numpy
import cPickle as pkl

import sys

def main(dict_path, emb_path, dim):
    with open(dict_path, 'r') as f:
       word_dict = pkl.load(f)
    num = len(word_dict)
    emb = 0.01 * numpy.random.randn(num, dim).astype('float32')
    count = 0
    with open(emb_path, 'r') as f:
        for line in f:
            l = line.strip().split(' ')
            if l[0] in word_dict:
                emb[word_dict[l[0]], :] = l[1:]
                count += 1
    print 'oov num:', num - 3 - count
    save_name = 'snli_emb_%i' % dim
    with open('%s.pkl'%save_name, 'wb') as f:
        pkl.dump(emb, f)

    print 'Done'

if __name__ == '__main__':
    emb_path = './glove.840B.300d.txt'
    dict_path = './snli_dict.pkl'
    dim = 300
    main(dict_path, emb_path, dim)




