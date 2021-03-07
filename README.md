# AAMAS21-375
Code for the AAMAS_21 paper id 375 titled "Action Selection for Composable Modular Deep Reinforcement Learning". This repo assumed you have carefully read the full paper and the additional appendix.

The GRACIAS agent for Bunny world and racetrack is named HierarchicalDDPGAgent. For Qbert it is called GRACIAS.

## To run bunny world:
python-2.7.12

Install the requirements specified in [requirement.txt](https://github.com/damnOblivious/AAMAS21-375/blob/main/requirements.txt)

python pacman.py -p [AgentName] -x [TrainingEpisodes] -n [TotalEpisodes]  -l [GridName] -g [GhostAgent's Policy] -q
Example:
python -u pacman.py -p GmQ_Pre -x 2000 -n 2001  -l smallInitialGrid -g DirectionalGhost -q


Note: Use only this pair of grid and ghost agent.

Running agents with pre-trained modules:
- First save the modules using lines like 569 in bunny-world/qlearningAgents.py
- Choose one of the models from those saved in weights folder
- Change the name of the pre-trained model accordingly in the agent you want to run. Eg line 602 in bunny-world/qlearningAgents.py


## To run racetrack:
python-2.7.12

Install the requirements specified in [requirement.txt](https://github.com/damnOblivious/AAMAS21-375/blob/main/requirements.txt)

python racetrack.py
To select specific agent change line #35 of racetrack/racetrack.py

Running agents with pre-trained modules: Follow steps similar to the bunny world.

## To run qbert:
Install the requirements specified in [requirement.txt](https://github.com/damnOblivious/AAMAS21-375/blob/main/qbert/requirements.txt)

python3 examples.py
To select specific agent change line #489 of qbert/examples.py
