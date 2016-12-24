from collections import OrderedDict 
import cPickle as pkl 
import sys
import fileinput
def main():
    bk_dict = OrderedDict()
    for filename in sys.argv[1:]:
        print filename
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                bk_dict[line[1]] = line[0]
    save_name = 'bk_dict'
    with open('%s.pkl'%save_name, 'wb') as f:
        pkl.dump(bk_dict, f)

if __name__ == '__main__':
    main()
