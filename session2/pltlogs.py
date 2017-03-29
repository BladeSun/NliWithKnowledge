import sys

f=open(sys.argv[1],'r')
step = []
valid_acc = []
test_acc = []
for line in f.readlines():
    line = line.split(' ')
    print line[9]
    if line[9] == 'ValidAcc:':
        step.append(int(line[8]))
        valid_acc.append(float(line[10]))
        test_acc.append(float(line[12]))

print step
print valid_acc
print test_acc
    
