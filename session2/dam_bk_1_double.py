'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time
import logging

from collections import OrderedDict

from data_iterator_bk import TextIterator

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'funcf_layer': ('param_init_funcf_layer', 'funcf_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def relu(x):
    return tensor.nnet.relu(x)

def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation
def prepare_data(seqs_x, seqs_x_syn, seqs_y, seqs_y_syn, label, maxlen=None, n_words_src=30000,
                 n_words=30000, bk_for_x=None, bk_for_y=None, bk_dim=10):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_x_syn = []
        new_seqs_y = []
        new_seqs_y_syn = []
        new_lengths_x = []
        new_lengths_y = []
        new_label = []
        for l_x, s_x, s_x_syn, l_y, s_y, s_y_syn, ll in zip(lengths_x, seqs_x, seqs_x_syn, lengths_y, seqs_y, seqs_y_syn, label):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_seqs_x_syn.append(s_x_syn)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_seqs_y_syn.append(s_y_syn)
                new_lengths_y.append(l_y)
                new_label.append(ll)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        seqs_x_syn = new_seqs_x_syn
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        seqs_y_syn = new_seqs_y_syn
        label = new_label

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 2
    maxlen_y = numpy.max(lengths_y) + 2

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    x_syn = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    y_syn = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    flabel = numpy.array(label).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_x_syn, s_y, s_y_syn] in enumerate(zip(seqs_x, seqs_x_syn, seqs_y, seqs_y_syn)):
        x[0, idx] = 1
        x[lengths_x[idx]+1, idx] = 2
        x[1:lengths_x[idx] + 1, idx] = s_x
        x_mask[:lengths_x[idx] + 2, idx] = 1.
        
        x_syn[0, idx] = 3 # 3 for none
        x_syn[lengths_x[idx]+1, idx] = 3
        x_syn[1:lengths_x[idx] + 1, idx] = s_x_syn


        y[0, idx] = 1
        y[lengths_y[idx]+1, idx] = 2
        y[1:lengths_y[idx] + 1, idx] = s_y
        y_mask[:lengths_y[idx] + 2, idx] = 1.

        y_syn[0, idx] = 3 # 3 for none
        y_syn[lengths_y[idx]+1, idx] = 3
        y_syn[1:lengths_y[idx] + 1, idx] = s_y_syn


    getbk = lambda sid, batch_id, target, bkdict: numpy.array([numpy.array(bkdict[sid][tid]).astype('float32') if tid in bkdict[sid] else numpy.zeros(bk_dim).astype('float32') for tid in target[:, batch_id]])
    bk_x = numpy.array([getbk(z[0], z[1], y_syn, bk_for_x) if z[0] in bk_for_x else numpy.zeros((maxlen_y,bk_dim)).astype('float32') for z in zip(x_syn.reshape(-1).tolist(), range(n_samples) * maxlen_x) ]).reshape(maxlen_x, n_samples, maxlen_y, bk_dim)
    bk_y = numpy.array([getbk(z[0], z[1], x_syn, bk_for_y) if z[0] in bk_for_y else numpy.zeros((maxlen_x,bk_dim)).astype('float32') for z in zip(y_syn.reshape(-1).tolist(), range(n_samples) * maxlen_y) ]).reshape(maxlen_y, n_samples, maxlen_x, bk_dim)

    bk_x = bk_x[:,:,:,(0,11,12)]
    return x, x_mask, bk_x, y, y_mask, bk_y, flabel

    

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# functionF layer
def param_init_funcf_layer(options, params, prefix='funcF', nin=None, nout=None,
                           ortho=True):
    if nin is None:
        nin = options['dim_word']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W1')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b1')] = numpy.zeros((nout,)).astype('float32')
    params[_p(prefix, 'W2')] = norm_weight(nout, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b2')] = numpy.zeros((nout,)).astype('float32')

    return params


def funcf_layer(tparams, state_below, options, prefix='funcF',
                activ='lambda x: tensor.tanh(x)', **kwargs):
    emb_proj = (tensor.dot(state_below, tparams[_p(prefix, 'W1')]) +
                tparams[_p(prefix, 'b1')])
    return eval(activ)(
        tensor.dot(emb_proj, tparams[_p(prefix, 'W2')]) +
        tparams[_p(prefix, 'b2')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
                   tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
                   tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim * 2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   **kwargs):
    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
            tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
                   tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
                   tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        # pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1, U_nl) + b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl) + bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    # seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context] + shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


'''

def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params
'''


def init_params(options):
    params = OrderedDict()

    # embedding
    #params['Wemb'] = norm_weight(options['dict_size'], options['dim_word'])
    params['Wemb'] = options['allembs']
    params['op_weights'] = norm_weight(options['op_num'] * options['op_dim'], options['op_dim'])
    params['op_V'] = numpy.random.randn(options['op_num']).astype('float32')

    params['bkW_input'] = norm_weight(options['dim_word'] , 1)
    params['bkW_op'] = norm_weight(options['op_dim'] , 1)
    # params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])
    params = get_layer('ff')[0](options, params,
                                         prefix='projOp',
                                         nin=options['dim_word'],
                                         nout=options['op_dim'])

    # funcf
    #params = get_layer('funcf_layer')[0](options, params,
    #                                     prefix='funcf',
    #                                     nin=options['dim_word'],
    #                                     nout=options['dim'])
    # funcG
    #params = get_layer('funcf_layer')[0](options, params,
    #                                     prefix='funcG',
    #                                     nin=options['dim_word'] * 2,
    #                                     nout=options['dim'])
    #params = get_layer('ff')[0](options, params, prefix='bkProj',
    #                            nin=options['dim'] + options['bk_dim'], nout=options['dim'],
    #                            ortho=False)
    #params = get_layer('ff')[0](options, params, prefix='WeightW',
    #                            nin=options['bk_dim'], nout=1,
    #                            ortho=False)
    params = get_layer('ff')[0](options, params, prefix='funcG',
                                nin=options['dim'] * 2, nout=options['dim'],
                                ortho=False)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim'] * 2, nout=options['dim'],
                                ortho=False)

    params = get_layer('ff')[0](options, params, prefix='ff_logit_linear',
                                nin=options['dim'], nout=options['class_num'],
                                ortho=False)

    return params


def build_dam(tparams, options):

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    bk_x = tensor.tensor4('x_bk', dtype='float32')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    bk_y = tensor.tensor4('y_bk', dtype='float32')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    #all_embs = tensor.matrix('emb', dtype='float32')
    label = tensor.vector('label', dtype='int64')

    n_timesteps_h = x.shape[0]
    n_timesteps_t = y.shape[0]
    n_samples = x.shape[1]

    emb_h = tparams['Wemb'][x.flatten()]
    emb_h = emb_h.reshape([n_timesteps_h, n_samples, options['dim_word']])
    if options['use_dropout']:
        emb_h = dropout_layer(emb_h, use_noise, trng)

    emb_t = tparams['Wemb'][y.flatten()]
    emb_t = emb_t.reshape([n_timesteps_t, n_samples, options['dim_word']])
    if options['use_dropout']:
        emb_t = dropout_layer(emb_t, use_noise, trng)

    #proj_h = get_layer('funcf_layer')[1](tparams, emb_h, options,
    #                                     prefix='funcf')
    #proj_t = get_layer('funcf_layer')[1](tparams, emb_t, options,
    #                                     prefix='funcf')
    weight_matrix = tensor.batched_dot(emb_h.dimshuffle(1, 0, 2), emb_t.dimshuffle(1, 2, 0))


    # bk_x
    bk_x = bk_x.dimshuffle(1,0,2,3)
    #bk_x = bk_x[:,:,:,(0,1,11,12)]
    bk_m = theano.tensor.repeat(bk_x, repeats=options['op_dim'], axis=3)
    bk_op = bk_m[:,:,:,:,None] * tparams['op_weights'][None,None,None,None,:,:]

    bk_op = bk_op.reshape([n_samples, n_timesteps_h, n_timesteps_t, options['op_num'] * options['op_dim'],options['op_dim']])
    bk_op = bk_op.dimshuffle(0,1,2,4,3)
    bk_op = bk_op.reshape([-1, options['op_dim'], options['op_num'] * options['op_dim']])

    emb_h_tmp = emb_h.dimshuffle(1,0,'x',2) + tensor.zeros([n_samples,n_timesteps_h,n_timesteps_t,options['dim']])
    emb_h_tmp = emb_h_tmp.reshape([-1, options['dim_word']])
    emb_h_tmpp = get_layer('ff')[1](tparams, emb_h_tmp, options,prefix='projOp', activ='relu')
    bk_hop = tensor.batched_dot(emb_h_tmpp, bk_op)

    #emb_h_tmp.dimshuffle(0, 'x', 1) * r_hop.reshape [-1, options['op_num'], options['dim']
    #bk_op = tensor.batched_dot(emb_h_tmp, bk_op)

    emb_t_tmp = emb_t.dimshuffle(1,'x',0,2) + tensor.zeros([n_samples,n_timesteps_h,n_timesteps_t,options['dim']])
    emb_t_tmp = emb_t_tmp.reshape([-1, options['dim_word']])
    emb_t_tmpp = get_layer('ff')[1](tparams, emb_t_tmp, options,prefix='projOp', activ='relu')
    bk_top = tensor.batched_dot(emb_t_tmpp, bk_op)

    #bk_hop = tensor.batched_dot(emb_h_tmp, bk_op)

    #weight_bk = (bk_op.reshape([-1, options['op_num'], options['op_dim']]) * emb_t_tmp.dimshuffle(0, 'x', 1)).sum(2)
    weight_bk = (bk_hop.reshape([-1, options['op_num'], options['op_dim']]) * bk_top.reshape([-1, options['op_num'], options['op_dim']])).sum(2)
    weight_bk = tensor.dot(tparams['op_V'], weight_bk.T)
    
    g_h =tensor.dot(emb_h_tmp , tparams['bkW_input'])
    g_t =tensor.dot(emb_t_tmp , tparams['bkW_input'])
    g_h_op =tensor.dot(emb_h_tmpp , tparams['bkW_op'])
    g_t_op =tensor.dot(emb_t_tmpp , tparams['bkW_op'])
    gate = tensor.nnet.sigmoid(g_h + g_t + g_h_op + g_t_op)

    weight_bk = weight_bk * gate.T

    weight_matrix = weight_matrix + weight_bk.reshape([n_samples, n_timesteps_h, n_timesteps_t])

    weight_matrix_1 = tensor.exp(weight_matrix - weight_matrix.max(1, keepdims=True)).dimshuffle(1,2,0)
    weight_matrix_2 = tensor.exp(weight_matrix - weight_matrix.max(2, keepdims=True)).dimshuffle(1,2,0)


    #  lenH * lenT * batchSize
    alpha_weight = weight_matrix_1 * x_mask.dimshuffle(0, 'x', 1)/ weight_matrix_1.sum(0, keepdims=True)
    beta_weight = weight_matrix_2 * y_mask.dimshuffle('x', 0, 1)/ weight_matrix_2.sum(1, keepdims=True)

    ##bk_y = bk_y.dimshuffle(2, 0, 1, 3)
    #emb_h_bk = theano.tensor.repeat(emb_h[:,None,:,:],repeats=n_timesteps_t, axis=1)  
    #emb_h_bk = theano.tensor.concatenate([emb_h_bk,bk_y.dimshuffle(2,0,1,3)], axis=3)
    #emb_h_bk = get_layer('ff')[1](tparams, emb_h_bk, options,prefix='bkProj', activ='relu')

    ## lenH * lenT * bachSize * dim
    ##bk_x = bk_x.dimshuffle(0, 2, 1, 3)
    #emb_t_bk = theano.tensor.repeat(emb_t[None,:,:,:],repeats=n_timesteps_h, axis=0)  
    #emb_t_bk = concatenate([emb_t_bk,bk_x.dimshuffle(0,2,1,3)], axis=3)
    #emb_t_bk = get_layer('ff')[1](tparams, emb_t_bk, options,prefix='bkProj', activ='relu')

    alpha = (emb_h.dimshuffle(0, 'x', 1, 2) * alpha_weight.dimshuffle(0, 1, 2, 'x')).sum(0)
    beta = (emb_t.dimshuffle('x', 0, 1, 2) * beta_weight.dimshuffle(0, 1, 2, 'x')).sum(1)
    #alpha = (emb_h_bk * alpha_weight.dimshuffle(0, 1, 2, 'x')).sum(0)
    #beta = (emb_t_bk * beta_weight.dimshuffle(0, 1, 2, 'x')).sum(1)

    v1 = concatenate([emb_h, beta], axis=2)
    v2 = concatenate([emb_t, alpha], axis=2)

    proj_v1 = get_layer('ff')[1](tparams, v1, options,prefix='funcG', activ='relu')
    proj_v2 = get_layer('ff')[1](tparams, v2, options, prefix='funcG', activ='relu')

    logit1 = (proj_v1 * x_mask[:, :, None]).sum(0)
    logit2 = (proj_v2 * y_mask[:, :, None]).sum(0)

    logit = concatenate([logit1, logit2], axis=1)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)

    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='tanh')
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit_linear', activ='linear')

    probs = tensor.nnet.softmax(logit)
    predict_label = probs.argmax(axis=1 )

    #cost = -tensor.log(probs)[tensor.arange(label.shape[0]), label]
    cost = tensor.nnet.categorical_crossentropy(probs, label)

    return trng, use_noise, x, x_mask, bk_x, y, y_mask, bk_y, label, predict_label, cost


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    # word embedding for backward rnn (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost


# build a sampler
def build_sampler(tparams, options, trng, use_noise):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder')
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r')

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):
    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score -= numpy.log(next_p[0, nw])
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k - dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=False):
    probs = []

    n_done = 0
    correct_num = 0
    all_num = 0.

    for x, x_syn, y, y_syn, label in iterator:
        n_done += len(x)
        all_num += len(label)

        x, x_mask, bk_x, y, y_mask, bk_y, label = prepare_data(x, x_syn, y, y_syn, label, 
                                                n_words_src=options['n_words_src'], bk_for_x=options['bk_for_x'], 
                                                bk_for_y=options['bk_for_y'], bk_dim=options['bk_dim'],
                                               maxlen= options['maxlen'],n_words=options['n_words'])

        pprobs, predict_label = f_log_probs(x, x_mask, bk_x, y, y_mask, bk_y, label)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >> sys.stderr, '%d samples computed' % (n_done)

        correct_num += (label == predict_label).sum() 

    print 'correct ', correct_num, 'all ', all_num 
    return numpy.array(probs), correct_num/all_num


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    updates = []

    t_prev = theano.shared(numpy.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2 ** t) / (1. - beta1 ** t)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g ** 2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))
    upreturn = [ item for sublist in updates for item in sublist]

    f_update = theano.function([lr], upreturn, updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    print 'adadelta'
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup + rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup + rgup + rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          bk_dim=13,  
          class_num=3,
          op_num=3,
          op_dim=50,
          encoder='gru',
          decoder='gru_cond',
          patience=1000000,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=100000,  # source vocabulary size
          n_words=100000,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='modelOpDouble.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq
          train_datasets=[
              '../data/train_h_fix.tok',
              '../data/train_t_fix.tok',
              '../data/train_label.tok',
              '../data/train_syn_h.syntok',
              '../data/train_syn_t.syntok'],
          valid_datasets=[
              '../data/dev_h_fix.tok',
              '../data/dev_t_fix.tok',
              '../data/dev_label.tok',
              '../data/dev_syn_h.syntok',
              '../data/dev_syn_t.syntok'],
          test_datasets=[
              '../data/test_h_fix.tok',
              '../data/test_t_fix.tok',
              '../data/test_label.tok',
              '../data/test_syn_h.syntok',
              '../data/test_syn_t.syntok'],
          dictionaries=[
              '../data/snli_dict_fix.pkl',
              '../data/bk_dict.pkl'],
          embedings=[
              '../data/snli_emb_300_fix.pkl'],
          bk_dicts=[
              '../data/bk_for_x.pkl',
              '../data/bk_for_y.pkl'],
          use_dropout=False,
          reload_=False,
          overwrite=False):
    # Model options
    model_options = locals().copy()
    log = logging.getLogger(os.path.basename(__file__).split('.')[0])

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    print 'Loading embedings ...'
    with open(embedings[0], 'rb') as f:
        pretrained_embs = pkl.load(f)
        #pretrained_embs = theano.shared(pretrained_embs, name='pretrained_embs')
    print 'Done'
    model_options['allembs'] = pretrained_embs

    print 'Loading bks ...'
    with open(bk_dicts[0], 'rb') as f:
        bk_for_x = pkl.load(f)
    model_options['bk_for_x'] = bk_for_x
    with open(bk_dicts[1], 'rb') as f:
        bk_for_y = pkl.load(f)
    model_options['bk_for_y'] = bk_for_x

    print 'Done'

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(train_datasets[0], train_datasets[1],
                         train_datasets[2], train_datasets[3], train_datasets[4],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_datasets[0], valid_datasets[1],
                         valid_datasets[2],valid_datasets[3],valid_datasets[4],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)
    test = TextIterator(test_datasets[0], test_datasets[1],
                        test_datasets[2], test_datasets[3], test_datasets[4],
                        dictionaries[0], dictionaries[1],
                        n_words_source=n_words_src, n_words_target=n_words,
                        batch_size=valid_batch_size,
                        maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto, params)

    tparams = init_tparams(params)


    trng, use_noise, \
    x, x_mask, bk_x, y, y_mask, bk_y, label, predict_label, \
    cost = \
        build_dam(tparams, model_options)
    inps = [x, x_mask, bk_x, y, y_mask, bk_y, label]

    # print 'Building sampler'
    # f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, [cost, predict_label], profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    # if decay_c > 0.:
    #     decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
    #     weight_decay = 0.
    #     for kk, vv in tparams.iteritems():
    #         weight_decay += (vv ** 2).sum()
    #     weight_decay *= decay_c
    #     cost += weight_decay

    ## regularize the alpha weights
    #if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
    #    alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
    #    alpha_reg = alpha_c * (
    #        (tensor.cast(y_mask.sum(0) // x_mask.sum(0), 'float32')[:, None] -
    #         opt_ret['dec_alphas'].sum(0)) ** 2).sum(1).mean()
    #    cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g ** 2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c ** 2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    bad_counter_acc = 0
    uidx = 0
    estop = False
    history_errs = []
    history_accs = []
    epoch_accs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size
    #if sampleFreq == -1:
    #    sampleFreq = len(train[0]) / batch_size

    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, x_syn, y, y_syn, label in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)
            try:
                x, x_mask, bk_x, y, y_mask, bk_y, label = prepare_data(x, x_syn, y, y_syn, label, maxlen=maxlen,
                                                n_words_src=n_words_src, bk_for_x=model_options['bk_for_x'], 
                                                bk_for_y=model_options['bk_for_y'], bk_dim=model_options['bk_dim'],
                                                n_words=n_words)
            
            except ValueError:
                print prepare_data(x, x_syn, y, y_syn, label, maxlen=maxlen,
                                                n_words_src=n_words_src, bk_for_x=model_options['bk_for_x'], 
                                                bk_for_y=model_options['bk_for_y'], bk_dim=model_options['bk_dim'],
                                                n_words=n_words)
                raise

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, bk_x, y, y_mask, bk_y, label)

            # do the update on parameters
            #print 'Befor:'
            #print tparams['ff_logit_W'].get_value()
            f_update(lrate)
            #print 'After:'
            #print tparams['ff_logit_W'].get_value()
            #update = f_update(lrate)
            #print update

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                log.info('Epoch: %d Update: %d Cost: %f UD: %f'%(eidx, uidx, cost, ud))

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip(tparams))
                    print 'Done'

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                #print 'Here:'
                #print tparams['ff_logit_W'].get_value()
                #print unzip(tparams)
                valid_errs, valid_acc = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                test_errs, test_acc = pred_probs(f_log_probs, prepare_data,
                                                 model_options, test)
                test_err = test_errs.mean()
                history_accs.append(test_acc)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        #estop = True
                        #break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                log.info('Epoch: %d Update: %d ValidAcc: %f TestAcc: %f' % (eidx, uidx, valid_acc, test_acc))

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples
        if len(history_accs) > 0:
            epoch_accs.append(history_accs[-1])
        if len(epoch_accs) > 1 and epoch_accs[-1] <= numpy.array(epoch_accs)[:-1].max():
            bad_counter_acc += 1
            if bad_counter_acc > 2:
                print 'Early Stop Acc!'
                #estop = True
                #break

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    test_err, acc = pred_probs(f_log_probs, prepare_data,
                           model_options, test)

    print 'Test acc ', acc 

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
