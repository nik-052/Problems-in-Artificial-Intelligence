import re
from itertools import permutations
from collections import deque
from sys import argv, exit
from time import perf_counter
from copy import deepcopy

def openProblem():

    objects = ['E', 'G', 'C', 'D', 'F', 'A', 'B']
    initTemp = [('CLEAR', 'B', ''), ('CLEAR', 'A', ''), ('ONTABLE', 'F', ''), ('ONTABLE', 'D', ''),
                ('ON', 'B', 'C'), ('ON', 'C', 'G'), ('ON', 'G', 'E'), ('ON', 'E', 'F'), ('ON', 'A', 'D')]
    goalTemp = [('E', 'B'), ('B', 'F'), ('F', 'D'),
                ('D', 'A'), ('A', 'C'), ('C', 'G')]

    init = {i: ['table', True] for i in objects}

    print('Init Temp', initTemp)
    print('Goal Temp', goalTemp)
    print('objects', objects)

    # print()

    for item in initTemp:
        if item[0] == 'ON':
            init[item[1]][0] = item[2]
            init[item[2]][1] = False
        # unessecary, but left for readability
        # elif item[0] == 'ONTABLE':
        #    state[item[1]][0] = 'table'
        # elif item[0] == 'CLEAR':
        #    state[item[1]][1] = True
        #####################################

    # Initialize goal and their state (position, is clear).
    goal = {i: ['table', True] for i in objects}

    # For each item that's is on another, change it's location
    # and set to unclear.
    for item in goalTemp:
        goal[item[0]][0] = item[1]
        goal[item[1]][1] = False

    return init, goal, objects

def writeSolution(solution):
    print('\n')
    i = 0
    for move in solution:
        i += 1
        print('{}. Move({}, {}, {})\n' .format(
            i, move[0], move[1], move[2]))

class State(object):
    """
        description
            A state's description dictionary looks like this...
            {'A': ['B', True], 'C': ['table', True], 'B': ['table', False]}

            And represents...
            'That cube': ['is on top of that', is it clear on top?]

            And if we visualize it, it looks like this...
             ___
            | A |
            |___|  ___
            | B | | C |
            |___| |___|
            ====================== <-- table

        parent
            The parent state object.

        move
            The move that was required to form that state from parent state.
            The list has the following format...
            ['A', 'B', 'C'] or ['A', 'B', 'table']
            ...which means, move cube A, from cube, on top of cube C or table.
    """

    def __init__(self, description=None, parent=None, move=None):
        super(State, self).__init__()
        self._parent = parent
        self._moveToForm = move

        # If no initial state description is given, try to create it,
        # by following the move that this state was given to form itself.
        if not description:
            self._stateDescription = deepcopy(self._parent._stateDescription)

            # If that move doesn't exists, it probably means, that it's a root state.
            if self._moveToForm is not None:
                self.__move(self._moveToForm[0], self._moveToForm[2])
        # Otherwise, just use the state given as argument.
        else:
            self._stateDescription = description

    # Overriding the equals method, so the comparison of the states is its
    # description dictionary.
    def __eq__(self, other):
        if other is None:
            return False
        return self._stateDescription == other._stateDescription

    # Overriding the representation method, for debugging purposes.
    def __repr__(self):
        return str(self._stateDescription) + '\n'

    def _generateStateChildren(self):
        """
            Generates all possible children (states) of itself (state).
            Each child state represents a possible move.
        """
        # Find all clear cubes of the state.
        clearCubes = [
            key for key in self._stateDescription if self._stateDescription[key][1] is True]

        # Calculate all possible move permutations and
        # add the special case of moving a cube onto table, if it's not already.
        possibleMoves = list(permutations(clearCubes, 2)) + [(
            cube, 'table') for cube in clearCubes if self._stateDescription[cube][0] != 'table']

        # Initialize the final generated children states list.
        states = []

        # For every possible move, create a child state, whose parent is this
        # very state and its move to form is given bt the move method.
        for cubeToMove, destinationCube in possibleMoves:
            states.append(State(parent=self, move=self.__move(
                cubeToMove, destinationCube, True)))

        return states

    def __move(self, object, destination, fake=False):
        """
            Moves the selected object to desired destination and
            returns the action in detail. Optionally,
            it only returns the hypothetical move, without actually doing it.
        """

        # Initialize the initial position of the cube.
        oldPosition = self._stateDescription[object][0]

        # Fake means, that the move is only recorded and not performed.
        # This is useful when we only want the move to form a state
        # from another and then passed as an argument to a new state object.
        if fake:
            return [object, oldPosition, destination]

        # The cube below is now clear, because the cube above it is lifted.
        # Unless it's the table, which is always something we can place on.
        if oldPosition != 'table':
            self._stateDescription[oldPosition][1] = True

        # Cube is now onto destination cube.
        self._stateDescription[object][0] = destination

        # The cube below is now unclear, because the cube above it is placed.
        # Unless it's the table, which is always something we can place on.
        if destination != 'table':
            self._stateDescription[destination][1] = False

        # [move a cube, from that cube, on top of another cube or on table]
        move = [object, oldPosition, destination]

        return move

    def __hash__(self):
        # Creating my own hashing method for the state, which is uniquely
        # identified by the cube, its position and a letter T(rue) or F(alse),
        # which denotes whether the cube is clear above or not.
        string = ''
        for key, value in self._stateDescription.items():
            string += "".join(key + value[0] + str(value[1])[0])
        return hash(string)

    def _tracePath(self):
        """
            Finds the moves required to solve the problem.
        """

        # Initialize the final path list.
        path = []
        # Initialize the current parent as this very state.
        currentParent = self

        # While there is a parent, we have not reached the root...
        while currentParent._parent is not None:
            # Add the move that the current parent required to form to the path.
            path.append(currentParent._moveToForm)
            # Set the current parent, the parent of the it, the grandparent.
            # So, we can go one vertex above, until there is no parent.
            # That means we have reached the root, because it has no parent.
            currentParent = currentParent._parent

        # Invert the order before you return, becase we need
        # the path from the root to the state,
        # but we have the path from this very state
        # (which is probably a solution) to the root.
        return path[::-1]

    def _tracePathDEBUG(self):
        # Just pretty printing the moves to solution.
        i = 0
        for move in self._tracePath():
            i += 1
            print('{}. Move({}, {}, {})' .format(i, move[0], move[1], move[2]))

def breadthFirstSearch(initialState, goalState, timeout=60):
    # Initialize iterations counter.
    iterations = 0

    # Initialize visited vertexes as set, because it's faster to check
    # if an item exists, due to O(1) searching complexity on average case.
    # The items here are hashable state objects.
    # A list, has O(n) on average case, when searching for an item existence.
    #
    # Initialize the search queue which is a double-ended queue and has O(1)
    # complexity on average case when popping an item from it's left.
    # A list has O(n) on average case, when popping from the left,
    # so a deque, improves performance for both ends accesses.
    #
    # source : https://wiki.python.org/moin/TimeComplexity
    visited, queue = set(), deque([initialState])

    # Initialize timeout counter.
    t1 = perf_counter()
    # While there are elements to search for...
    while queue:
        # Initialize on each iteration the performace of the previous.
        t2 = perf_counter()
        # If the the previous iteration has exceeded the allowed time,
        # then return, prematurely, nothing.
        if t2 - t1 > timeout:
            return None, iterations

        iterations += 1
        vertex = queue.popleft()

        if vertex == goalState:
            return vertex._tracePath(), iterations

        for neighbour in vertex._generateStateChildren():
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

def depthFirstSearch(initialState, goalState, timeout=60):
    # Initialize iterations counter.
    iterations = 0

    # Initialize visited vertexes as set, because it's faster to check
    # if an item exists, due to O(1) searching complexity on average case.
    # The items here are hashable state objects.
    # A list, has O(n) on average case, when searching for an item existence.
    #
    # Initialize the search queue which is a double-ended queue and has O(1)
    # complexity on average case when popping an item from it's right.
    # A list has O(1) on average case, when popping from the right,
    # which is the same, but we leave it the same as BFS for readability reasons.
    #
    # source : https://wiki.python.org/moin/TimeComplexity
    visited, stack = set(), deque([initialState])

    # Initialize timeout counter.
    t1 = perf_counter()

    # While there are elements to search for...
    while stack:
        # Initialize on each iteration the performace of the previous.
        t2 = perf_counter()
        # If the the previous iteration has exceeded the allowed time,
        # then return, prematurely, nothing.
        if t2 - t1 > timeout:
            return None, iterations

        iterations += 1
        vertex = stack.pop()  # right

        if vertex == goalState:
            return vertex._tracePath(), iterations

        if vertex in visited:
            continue

        for neighbour in vertex._generateStateChildren():
            stack.append(neighbour)

        visited.add(vertex)

def __distanceFromGoal(currentStates, goalState):
    """ The H function. """

    # Initialize a list of each state's scores.
    statesScores = []

    # For each state in currently discovered states...
    for state in currentStates:

        # Initialize out place blocks.
        outOfPlaceBlocks = 0

        # For each block in every state...
        for block in state._stateDescription:

            # If that block is not positioned correctly, increase out of place
            # blocks for that state.
            if state._stateDescription[block] != goalState._stateDescription[block]:
                outOfPlaceBlocks += 1

        # Store the final score for that state.
        statesScores.append(outOfPlaceBlocks)

    # Return the index of the state with smallest distance from goal.
    return statesScores.index(min(statesScores))

def __distanceFromGoalWithLeastMoves(currentStates, goalState):
    """ The G + H function. """

    # Initialize a list of each state's scores.
    statesScores = []

    # For each state in currently discovered states...
    for state in currentStates:

        # Initialize out place blocks.
        outOfPlaceBlocks = 0

        # For each block in every state...
        for block in state._stateDescription:

            # If that block is not positioned correctly, increase out of place
            # blocks for that state.
            if state._stateDescription[block] != goalState._stateDescription[block]:
                outOfPlaceBlocks += 1

        # Store how many blocks are out of place plus the number of moves
        # needed to reach from root to each state.
        statesScores.append(outOfPlaceBlocks + len(state._tracePath()))

    # Return the index of the state with smallest distance
    # and least moves from goal.
    return statesScores.index(min(statesScores))

def heuristicSearch(initialState, goalState, algorithm='best', timeout=60):
    # Each algorithm uses a different heuristic function for the search.
    if algorithm == 'astar':
        function = __distanceFromGoalWithLeastMoves
    elif algorithm == 'best':
        function = __distanceFromGoal

    # Initialize iterations counter.
    iterations = 0

    # Initialize visited vertexes as set, because it's faster to check
    # if an item exists, due to O(1) searching complexity on average case.
    # The items here are hashable state objects.
    # A list, has O(n) on average case, when searching for an item existence.
    #
    # Initialize the search list.
    # A list has O(n) for popping items on average case.
    # We cannot improve it any further, since we may access items in the middle.
    #
    # source : https://wiki.python.org/moin/TimeComplexity
    visited, list = set(), [initialState]

    # Initialize timeout counter.
    t1 = perf_counter()

    # While there are elements to search for...
    while list:
        # Initialize on each iteration the performace of the previous.
        t2 = perf_counter()
        # If the the previous iteration has exceeded the allowed time,
        # then return, prematurely, nothing.
        if t2 - t1 > timeout:
            return None, iterations

        iterations += 1
        # Determine which item you pop, defined by the heuristic function of
        # the corresponding algorithm.
        item = function(list, goalState)
        vertex = list.pop(item)

        if vertex == goalState:
            return vertex._tracePath(), iterations

        for neighbour in vertex._generateStateChildren():
            if neighbour in visited:
                continue

            visited.add(neighbour)
            list.append(neighbour)

############################

def main(argv):
    # if len(argv) > 4:
    #     print('Usage:\npython3 {} <algorithm> <problem_file_name.pddl> [solution_file_name]' .format(
    #         argv[0]))
    #     exit(1)

    # problemFile = argv[2]
    # outputFile = ''
    # if len(argv) == 4:
    #     outputFile = argv[3]

    init, goal, cubes = openProblem()

    # Initialize inital and goal states.
    initialState = State(init)
    goalState = State(goal)

    algorithm = 'astar'

    t1 = perf_counter()

    if algorithm == 'breadth':
        solution, iters = breadthFirstSearch(initialState, goalState)
    elif algorithm == 'depth':
        solution, iters = depthFirstSearch(initialState, goalState)
    elif algorithm == 'best' or algorithm == 'astar':
        solution, iters = heuristicSearch(initialState, goalState, algorithm)
    else:
        raise Exception(
            'Unknown algorithm. Available : breadth, depth, best, astar')

    t2 = perf_counter()

    # print('| Problem name: {}' .format(' ' * 10 + problemFile))
    print('| Algorithm used: {}' .format(' ' * 8 + algorithm))
    print('| Number of cubes: {}' .format(' ' * 7 + str(len(cubes))))
    print('| Cubes: {}' .format(' ' * 17 + str(' '.join(cubes))))
    if solution:
        print('| Solved in: {}' .format(' ' * 13 + str(t2-t1)))
        print('| Algorithm iterations: {}' .format(' ' * 2 + str(iters)))
        print('| Moves: {}' .format(' ' * 17 + str(len(solution))))

        print('| Solution:' + ' ' * 15 + 'Found!')
        writeSolution(solution)
    else:
        print('| Solution:' + ' ' * 15 + 'NOT found, search timed out.')

if __name__ == '__main__':
    main(argv)
