import mmath
import numpy as np

class NeturalCell():
    def __init__(self, level_num, cell_num, next_cellnums):
        """ cell: ##j
        """
        self.level_num = level_num
        self.cell_num = cell_num
        self.outval = 0
        self.inval = 0
        self.weights = []*next_cellnums
        self.bias = 0
class NeturalLevel():

    def __init__(self, level_num, cellnums):
        self.cells = [] * cellnums
        for i in cellnums:
            self.cells.append(NeturalCell(level_num, i , next_cellnums))
        
class NeturalNetwork():
    f = None
    def __init__(self, X, Y, level_param: list):

        self.levelsize = len(level_param) + 2
        self.levels = [] * len(self.levelsize)
        self.levels.append(NeturalLevel(0, X.shape[1]))
        for i, cellnums in enumerate(level_param):            
            self.levels.append(NeturalLevel(i + 1, cellnums))
        self.levels.append(NeturalLevel(self.levelsize - 1, Y.shape[1]))

    
    