import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os
from abc import abstractmethod


class Gradient:
    def __init__(self, opts: dict):
        self.opts = opts
