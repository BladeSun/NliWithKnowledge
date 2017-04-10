import numpy
import cPickle as pkl

import sys
import fileinput

from collections import OrderedDict

def main():
    word_freqs = OrderedDict()
    for filename in sys.argv[1:]:
        print 'Processing', filename
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
    words = word_freqs.keys()
    freqs = word_freqs.values()

    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()
    worddict['eos'] = 2
    worddict['b_o_s'] = 1
    worddict['pad_unit'] = 0 
    worddict['UNK'] = 3 
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii+4

    save_name = 'snli_dict_fix'
    with open('%s.pkl'%save_name, 'wb') as f:
        pkl.dump(worddict, f)

    print 'Done'

if __name__ == '__main__':
    main()
