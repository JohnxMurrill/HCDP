# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:34:15 2017

@author: murri101
"""

import json, collections
from pprint import pprint

def readInfile(data_file):
    with open(data_file) as f:    
        data = json.load(f)
    return [data[j:j+9] for j in xrange(0,len(data),9)][:100]
    
data = 'NineRound_inFile.txt'
out = readInfile(data)[:2]
for i in out:
    pad = [[[0,85,0],i[0][2][1]]] + [k[2] for k in i] + [[[10,0,0], 0]]
for i in range(len(pad)):
    pad[i][1] = pad[-2][1] - pad[i][1]

DPState = collections.namedtuple('DPState', 'period, health, cash')
pad = [[DPState(val[0][0],val[0][1],val[0][2]),val[1]] for val in pad]
pprint(pad)