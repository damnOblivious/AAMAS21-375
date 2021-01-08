''' color transition '''
def change_reward0(reward):
    if reward == 25:return 5
    if reward == 3100:return 20
    return 0

''' disc module '''
def change_reward1(reward):
    if reward == 500:return 10
    return 0

''' green module '''
def change_reward2(reward):
    if reward == 100:return 10
    if reward == 300:return 10
    return 0

''' life module'''
def change_reward3(reward, done):
    if reward == 0 and done:return -10
    return 1

