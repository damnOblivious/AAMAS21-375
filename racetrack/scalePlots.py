import matplotlib.pyplot as plt
from ddpgTrials import *
from gmTrials import *
def runningAvg(mylist, windowSize = 10):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):

        cumsum.append(cumsum[i-1] + x)
        if i < windowSize:
            moving_ave = cumsum[i] / (i)
        else:
            moving_ave = (cumsum[i] - cumsum[i-windowSize])/windowSize
        moving_aves.append(moving_ave)

    return moving_aves

def averageVector(vectors):
    vecNum = len(vectors)
    average = []
    min = len(vectors[0])
    for vec in vectors:
        if min > len(vec): min = len(vec)

    for i in range(min):
        sum = .0
        for j in range(vecNum):
            sum += vectors[j][i]
        average.append( sum / vecNum )

    return average

# plt.plot(runningAvg(averageVector([Gm17_1, Gm17_2, Gm17_3, Gm17_4]), 20), color='b', label = 'gm3')
# plt.plot(runningAvg(averageVector([ddpg17_1, ddpg17_2, ddpg17_3, ddpg17_4]), 20), color='k', label = 'ddpg3')

# plt.plot(runningAvg(averageVector([Gm17_1]), 20), color='b', label = 'ddpg3')
# plt.plot(runningAvg(averageVector([Gm17_2]), 20), color='b', label = 'ddpg3')
# plt.plot(runningAvg(averageVector([Gm17_3]), 20), color='b', label = 'ddpg3')
# plt.plot(runningAvg(averageVector([Gm17_4]), 20), color='b', label = 'ddpg3')

plt.plot(runningAvg(averageVector([ddpg14_1]), 20), color='k', label = 'ddpg3')
plt.plot(runningAvg(averageVector([ddpg14_2]), 20), color='k', label = 'ddpg3')
plt.plot(runningAvg(averageVector([ddpg14_3]), 20), color='k', label = 'ddpg3')
plt.plot(runningAvg(averageVector([ddpg14_4]), 20), color='k', label = 'ddpg3')

plt.legend(bbox_to_anchor=(0.95,0.2), loc=1, borderaxespad=0.)
plt.xlabel('Number of training epochs')
plt.ylabel('Average game score')

# plt.savefig("img/Qbert_nature.png",bbox_inches='tight')

plt.show()
