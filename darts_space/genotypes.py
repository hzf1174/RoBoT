from collections import namedtuple
import numpy as np
from copy import deepcopy

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

CIFAR_10_arch = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 1),
                            ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3)],
                    normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2),
                                                       ('skip_connect', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0),
                                                       ('sep_conv_5x5', 4), ('skip_connect', 3)],
                    reduce_concat=range(2, 6))

CIFAR_100_arch = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0),
                             ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('skip_connect', 1), ('sep_conv_3x3', 4)],
                     normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2),
                                                        ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 3),
                                                        ('avg_pool_3x3', 1), ('avg_pool_3x3', 4)],
                     reduce_concat=range(2, 6))

ImageNet_arch = Genotype(normal=[('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0),
                            ('sep_conv_5x5', 3), ('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
                    normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 0),
                                                       ('dil_conv_3x3', 2), ('max_pool_3x3', 2), ('skip_connect', 3),
                                                       ('sep_conv_3x3', 3), ('sep_conv_3x3', 0)],
                    reduce_concat=range(2, 6))
