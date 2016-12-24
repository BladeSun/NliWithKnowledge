__author__ = 'Suneburg'
import numpy
import cPickle as pkl

import sys

def main(bk_path):
    bk_for_x = {}
    bk_for_y = {}
    with open(bk_path, 'r') as f:
        for line in f:
            l = line.strip().split(' ')
            ids = l[0].split('_')
            if int(ids[0]) in bk_for_x:
                bk_for_x[int(ids[0])][int(ids[1])] = map(float, l[1:])
            else:
                bk_for_x[int(ids[0])] = {int(ids[1]):map(float, l[1:])} 
            if int(ids[1]) in bk_for_y:
                bk_for_y[int(ids[1])][int(ids[0])] = map(float, l[1:])
            else:
                bk_for_y[int(ids[1])] = {int(ids[0]):map(float, l[1:])} 
    save_name = 'bk_for_x'
    with open('%s.pkl'%save_name, 'wb') as f:
        pkl.dump(bk_for_x, f)
    save_name = 'bk_for_y'
    with open('%s.pkl'%save_name, 'wb') as f:
        pkl.dump(bk_for_y, f)

    print 'Done'

if __name__ == '__main__':
    bk_path = './bk_vec_new.txt'
    main(bk_path)

