# AAMAS21-375
Code link to the AAMAS_21 paper id 375 titled "Action Selection for Composable Modular Deep Reinforcement Learning".

## To run bunny world:
python pacman.py -p [AgentName] -x [TrainingEpisodes] -n [TotalEpisodes]  -l [GridName] -g [GhostAgent's Policy] -q
Example:
python -u pacman.py -p GmQ_Pre -x 2000 -n 2001  -l smallInitialGrid -g DirectionalGhost -q

## To run racetrack:
python racetrack.py
To select specific agent change line #35 of racetrack/racetrack.py

## To run qbert:
python3 examples.py
To select specific agent change line #489 of qbert/examples.py
