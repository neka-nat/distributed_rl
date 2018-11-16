import sys
import random
from collections import namedtuple
import numpy as np
from PIL import Image
if sys.version_info.major == 3:
    import _pickle as cPickle
else:
    import cPickle
import torch
_USE_COMPRESS = True
if _USE_COMPRESS:
    import lz4.frame

class Transition(namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))):
    __slots__ = ()
    def __new__(cls, state, action, reward, next_state=None, done=None):
        return super(Transition, cls).__new__(cls, state, action, reward, next_state, done)

_outsize = lambda x, f, p, s: int(x - f + 2 * p) / s + 1

def outsize(x, f, p=0, s=1):
    return (_outsize(x[0], f, p, s), _outsize(x[1], f, p, s))

def preprocess(img, shape=None, gray=False):
    pil_img = Image.fromarray(img)
    if not shape is None:
        pil_img = pil_img.resize(shape)
    if gray:
        img_ary = np.asarray(pil_img.convert("L"))
    else:
        img_ary = np.asarray(pil_img).transpose((2, 0, 1))
    return np.ascontiguousarray(img_ary, dtype=np.float32) / 255

def epsilon_greedy(state, policy_net, eps=0.1):
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1).cpu()
    else:
        return torch.tensor([random.randrange(policy_net.n_action)], dtype=torch.long)

def dumps(data):
    if _USE_COMPRESS:
        return lz4.frame.compress(cPickle.dumps(data))
    else:
        return cPickle.dumps(data)

def loads(packed):
    if _USE_COMPRESS:
        return cPickle.loads(lz4.frame.decompress(packed))
    else:
        return cPickle.loads(packed)

def rescale(x, eps=0.1):
    return x.sign() * ((x.abs() + 1.0).sqrt() - 1.0) + eps * x

def inv_rescale(x, eps=0.1):
    eps2 = 2.0 * eps
    eps21 = 2.0 * eps + 1.0
    xabs = x.abs()
    return x.sign() * (eps2 * xabs + eps21 - (2.0 * eps2 * xabs + eps21 * eps21).sqrt()) / (eps2 * eps)
