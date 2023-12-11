#!/usr/bin/env python
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))


class InvalidProblemException(Exception):
    pass


class NotConfiguredException(Exception):
    pass


class PathNotFoundException(Exception):
    pass
