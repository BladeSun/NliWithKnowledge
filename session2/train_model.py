import logging
import sys
import argparse
from time import gmtime, strftime
# set args
parser = argparse.ArgumentParser()
parser.add_argument('--model_name')
parser.add_argument('--dim_word', type=int)
parser.add_argument('--dim', type=int)
parser.add_argument('--optimizer')
parser.add_argument('--decay-c', type=float)
parser.add_argument('--clip-c', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--valid_batch_size', type=int)
parser.add_argument('--lrate', type=float)
parser.add_argument('--use_dropout', type=bool)
parser.add_argument('--validFreq', type=int)
parser.add_argument('--dispFreq', type=int)
parser.add_argument('--saveFreq', type=int)
parser.add_argument('--patience', type=int)
parser.add_argument('--maxlen', type=int)
allargs = vars(parser.parse_args())
# set logger
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(allargs['model_name'])
log.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

logfile = strftime("%Y%m%d_%H:%M:%S_", gmtime()) + allargs['model_name'] + '.log'
file_handler = logging.FileHandler('./logs/' + logfile, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

log.addHandler(console_handler)
log.addHandler(file_handler)

log.info('Enter Main Function, log file name is '+ logfile)
funcargs = {k:v for k, v in allargs.items() if v != None and k != 'model_name'}
model = __import__(allargs['model_name'])
model.train(**funcargs)
