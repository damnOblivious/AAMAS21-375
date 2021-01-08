import os

numTrials = 12

for i in range(numTrials):
    os.system('python -u racetrack.py ' + str(i))
