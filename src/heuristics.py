#!/usr/bin/env python

# Fix to include error TODO change it to something better
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))

import math
from custom_types import Node2d

def chebyshev(p1 : Node2d, p2 : Node2d) -> int:
    return max(abs(p1.x - p2.x), abs(p1.y - p2.y))

def euclidian(p1 : Node2d, p2 : Node2d) -> float:
    return(math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)) 


