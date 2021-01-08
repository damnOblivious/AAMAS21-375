import math
import numpy as np
import pickle

import layout_parser
from variables import *

layout_name = 'f1'
layout = layout_parser.getLayout( layout_name )

grid = layout.racetrack
dist = [[-1 for i in range(layout.height) ] for j in range(layout.width)]
q = []

for i in range(layout.width):
    for j in range(layout.height):
        if grid[i][j] == FINISH_CELL:
            q.append((i,j))
            dist[i][j] = 0

while len(q):
    x1, y1 = q.pop(0)

    if grid[x1-1][y1] != WALL_CELL and dist[x1-1][y1] == -1:
        q.append((x1-1,y1))
        dist[x1-1][y1] = dist[x1][y1] + 1

    if grid[x1+1][y1] != WALL_CELL and dist[x1+1][y1] == -1:
        q.append((x1+1,y1))
        dist[x1+1][y1] = dist[x1][y1] + 1

    if grid[x1][y1-1] != WALL_CELL and dist[x1][y1-1] == -1:
        q.append((x1,y1-1))
        dist[x1][y1-1] = dist[x1][y1] + 1

    if grid[x1][y1+1] != WALL_CELL and dist[x1][y1+1] == -1:
        q.append((x1,y1+1))
        dist[x1][y1+1] = dist[x1][y1] + 1

pickle.dump( dist, open( "dist.p", "wb" ) )
pickle.dump( grid, open( "grid.p", "wb" ) )
