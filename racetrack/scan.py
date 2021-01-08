import sys
totalLines = 0
for line in sys.stdin:
    totalLines += 1
    line = line.strip('\n')
    #line = line.split('[')
    line = line[16:]
    print 'ddpg'+str(totalLines + 11)+'_4 = ', line
