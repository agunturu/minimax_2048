from random import randint
from BaseAI import BaseAI
import math
import time

class PlayerAI(BaseAI):
    def __init__(self):
        self.possibleNewTiles = [2, 4]
        self.probability = 0.9
        self.timeLimit = 0.2

    def getNewTileValue(self):
        """
        Generate a new tile value with the same probablity as the
        GameManager.
        """
        if randint(0, 99) < 100 * self.probability:
            return self.possibleNewTiles[0]
        else:
            return self.possibleNewTiles[1]
    
    def emptyCellsHeuristic(self, grid):
        """
        This heuristic counts the number of empty cells. In general, a state with
        higher number of empty cells is preferred.
        """
        cells = grid.getAvailableCells()
        #return len(cells)
        return math.log(len(cells)) if cells else 0

    def maxTileHeuristic(self, grid):
        """
        This heuristic returns the maximum tile value. In general,
        a state with a high max tile is preferred.
        """
        #return math.log(grid.getMaxTile(), 2)
        return grid.getMaxTile()

    def monotonocityHeuristic(self, grid):
        """
        This heuristic measures how the values of tiles are monotonically 
        increasing or decreasing. Intuitively, this heuristic prefers states where
        higher valued tiles are at the edges.
        """
        up = 0
        down = 0
        right = 0
        left = 0

        # up/down monotonocity
        for x in xrange(grid.size):
            curr = 0
            nex = curr+1

            while nex < grid.size:
                while nex < grid.size:
                    if grid.getCellValue((x, nex)) != 0:
                        break
                    nex += 1

                if nex >= grid.size:
                    nex -= 1

                currVal = grid.getCellValue((x, curr))
                if currVal != 0:
                    currVal = math.log(currVal, 2)

                nexVal = grid.getCellValue((x, nex))
                if nexVal != 0:
                    nexVal = math.log(nexVal, 2)

                if currVal > nexVal:
                    up += nexVal - currVal
                else:
                    down += currVal - nexVal
                
                curr = nex
                nex += 1

        # left/right monotonocity
        for y in xrange(grid.size):
            curr = 0
            nex = curr+1

            while nex < grid.size:
                while nex < grid.size:
                    if grid.getCellValue((nex, y)) != 0:
                        break
                    nex += 1

                if nex >= grid.size:
                    nex -= 1

                currVal = grid.getCellValue((curr, y))
                if currVal != 0:
                    currVal = math.log(currVal, 2)

                nexVal = grid.getCellValue((nex, y))
                if nexVal != 0:
                    nexVal = math.log(nexVal, 2)

                if currVal > nexVal:
                    right += nexVal - currVal
                else:
                    left += currVal - nexVal
                
                curr = nex
                nex += 1

        return max(up, down) + max(right, left)

    def smoothnessHeuristic(self, grid):
        """
        This heuristic measures pair-wise differences of adjacent tiles. 
        Intuitively, this heuristic measures how easy it is to merge the
        adjacent tiles. A perfectly smooth state has a heuristic of 0 and
        other states have a negative heuristic value.
        """
        smoothness = 0
        for x in xrange(grid.size):
            for y in xrange(grid.size):
                cellVal = grid.getCellValue((x, y))
                if cellVal != 0:
                    val = math.log(cellVal, 2)

                    for k in xrange(x+1, grid.size):
                        targetCellVal = grid.getCellValue((k, y))
                        if targetCellVal is None:
                            break

                        if targetCellVal != 0:
                            targetVal = math.log(targetCellVal, 2)
                            smoothness -= math.fabs(val-targetVal)
                            break
                    
                    for k in xrange(y+1, grid.size):
                        targetCellVal = grid.getCellValue((x, k))
                        if targetCellVal is None:
                            break

                        if targetCellVal != 0:
                            targetVal = math.log(targetCellVal, 2)
                            smoothness -= math.fabs(val-targetVal)
                            break
                    
        return smoothness

    def evalfn(self, grid):
        """
        This function evalutes the utility of the current state of the grid.
        It uses different weights for different heuristics. 
        """
        smoothWeight = 0.1
        monoWeight = 1.0
        emptyWeight = 2.5
        maxWeight = 1.0

        return self.emptyCellsHeuristic(grid)*emptyWeight + \
                self.maxTileHeuristic(grid)*maxWeight + \
                self.monotonocityHeuristic(grid)*monoWeight + \
                self.smoothnessHeuristic(grid)*smoothWeight

    def minimize(self, grid, alpha, beta, depth, startTime):
        """
        This function minimizes the utility.
        """
        if depth == 0:
            return (grid, self.evalfn(grid))

        cells = grid.getAvailableCells()
        if not cells:
            return (None, self.evalfn(grid))

        (minChild, minUtility) = (None, float("inf"))
        
        for cell in cells:
            value = self.getNewTileValue()
            gridCopy = grid.clone()
            gridCopy.insertTile(cell, value)

            (temp, utility) = self.maximize(gridCopy, alpha, beta, depth-1, startTime)
            if utility < minUtility:
                (minChild, minUtility) = (gridCopy, utility)

            if minUtility <= alpha:
                break

            if minUtility < beta:
                beta = minUtility

        return (minChild, minUtility)
            
    def maximize(self, grid, alpha, beta, depth, startTime):
        """
        This function maximizes the utility.
        """
        if (depth == 0) or (time.clock() - startTime >= self.timeLimit):
            # return if max depth is reached or time limit exceeded..
            return (None, self.evalfn(grid))
            
        moves = grid.getAvailableMoves()

        if not moves:
            return (None, self.evalfn(grid))

        (bestMove, maxUtility) = (None, float("-inf"))

        for move in moves:
            gridCopy = grid.clone()
            gridCopy.move(move)
            (temp, utility) = self.minimize(gridCopy, alpha, beta, depth-1, startTime)

            if utility > maxUtility:
                (bestMove, maxUtility) = (move, utility)

            if maxUtility >= beta:
                break;

            if maxUtility > alpha:
                alpha = maxUtility

        return (bestMove, maxUtility)

    def decision(self, grid):
        depth = 0
        (bestMove, maxUtility) = (None, float("-inf"))
        startTime = time.clock()

        # employ an iterative deepening search with a time limit
        for depth in range(1, 16):
            if time.clock() - startTime >= self.timeLimit:
                break

            (move, utility) = self.maximize(grid, float("-inf"), float("inf"), depth, startTime)
            if utility > maxUtility:
                (bestMove, maxUtility) = (move, utility)

        return bestMove

    def getMove(self, grid):
        return self.decision(grid)
