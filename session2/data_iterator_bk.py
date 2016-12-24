import numpy

import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, label, source_synset, target_synset, 
                 all_dict, syn_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.source_synset = fopen(source_synset, 'r')
        self.target_synset = fopen(target_synset, 'r')
        self.label = fopen(label, 'r')
        with open(all_dict, 'rb') as f:
            self.all_dict = pkl.load(f)
        self.label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        with open(syn_dict, 'rb') as f:
            self.syn_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.source_buffer = []
        self.target_buffer = []
        self.source_synset_buffer = []
        self.target_synset_buffer = []
        self.label_buffer = []
        self.k = batch_size * 20

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.source_synset.seek(0)
        self.target_synset.seek(0)
        self.label.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        source_synset = []
        target_synset = []
        label = []


        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                ss_synset = self.source_synset.readline()
                if ss_synset == "":
                    break
                tt_synset = self.target_synset.readline()
                if tt_synset == "":
                    break
                ll = self.label.readline()
                if ll == "":
                    break

                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())
                self.source_synset_buffer.append(ss_synset.strip().split())
                self.target_synset_buffer.append(tt_synset.strip().split())
                self.label_buffer.append(ll.strip().split())

            # sort by target buffer
            tlen = numpy.array([len(t) for t in self.target_buffer])
            tidx = tlen.argsort()

            _sbuf = [self.source_buffer[i] for i in tidx]
            _tbuf = [self.target_buffer[i] for i in tidx]
            _sbuf_synset = [self.source_synset_buffer[i] for i in tidx]
            _tbuf_synset = [self.target_synset_buffer[i] for i in tidx]
            _lbuf = [self.label_buffer[i] for i in tidx]

            self.source_buffer = _sbuf
            self.target_buffer = _tbuf
            self.source_synset_buffer = _sbuf_synset
            self.target_synset_buffer = _tbuf_synset
            self.label_buffer = _lbuf

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0 or len(self.source_synset_buffer) == 0 or len(self.target_synset_buffer) == 0 or len(self.label_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.all_dict[w] if w in self.all_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 3 for w in ss]
                ss_syn = self.source_synset_buffer.pop()
                ss_syn = [self.syn_dict[w] if w in self.syn_dict else 3
                      for w in ss_syn]

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.all_dict[w] if w in self.all_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 3 for w in tt]
                tt_syn = self.target_synset_buffer.pop()
                tt_syn = [self.syn_dict[w] if w in self.syn_dict else 3 
                      for w in tt_syn]

                # get label
                ll = self.label_dict[self.label_buffer.pop()[0]]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                #need fix here!!!!
                if len(ss) != len(ss_syn) or len(tt) != len(tt_syn):
                    continue

                source.append(ss)
                source_synset.append(ss_syn)
                target.append(tt)
                target_synset.append(tt_syn)
                label.append(ll)

                #if len(ss) != len(ss_syn) or len(tt) != len(tt_syn):
                #    raise Exception, 'orign != syn'

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size or \
                        len(label) >= self.batch_size :
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, source_synset, target, target_synset, label
