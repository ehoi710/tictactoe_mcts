import numpy as np

class State:
    def __init__(self, board):
        self.board = board

    def get_legal_actions(self):
        actions = []
        for i, j in np.ndindex(3, 3):
            if self.board[i, j] == 0:
                actions.append((i, j))

        return actions

    def is_game_over(self):
        if self.game_result() != 0:
            return True

        for i, j in np.ndindex(3, 3):
            if self.board[i, j] == 0:
                return False
        
        return True

    def game_result(self):
        for i in range(3):
            if (self.board[i, 0] == self.board[i, 1] and 
                self.board[i, 1] == self.board[i, 2]):
                return self.board[i, 0]

            if (self.board[0, i] == self.board[1, i] and 
                self.board[1, i] == self.board[2, i]):
                return self.board[0, i]

        if (self.board[0, 0] == self.board[1, 1] and
            self.board[1, 1] == self.board[2, 2]):
            return self.board[0, 0]

        if (self.board[2, 0] == self.board[1, 1] and
            self.board[1, 1] == self.board[0, 2]):
            return self.board[2, 0]

        return 0

    def move(self, action, turn):
        copy = self.board.copy()
        copy[action] = turn

        return State(copy)

class Node:
    def __init__(self, state: State, parent=None, parent_action=None, color=1):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action

        self.color = color
        self.child = []

        self._visits = 0
        self._result = {
            1: 0,
            0: 0,
            -1: 0
        }

        self._untried = None
        self._untried = self.untried()

    def untried(self):
        self._untried = self.state.get_legal_actions()
        return self._untried

    def q(self):
        return self._result[-1] - self._result[1] + self._result[0]

    def n(self):
        return self._visits

    def expand(self):
        action = self._untried.pop()
        next_state = self.state.move(action, self.color)

        child_node = Node(next_state, self, action, -self.color)
        self.child.append(child_node)

        return child_node

    def is_game_over(self):
        return self.state.is_game_over()

    def rollout(self):
        current_state = self.state

        while not current_state.is_game_over():
            moves = current_state.get_legal_actions()
            action = self.rollout_policy(moves)

            current_state = current_state.move(action, self.color)

        return current_state.game_result()

    def backpropagate(self, result):
        self._visits += 1
        if result == self.color: self._result[1] += 1
        elif result == 0: self._result[0] += 1
        elif result != self.color: self._result[-1] += 1

        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried) == 0

    def best_child(self, c_p=.1):
        w = [(c.q() / c.n()) + c_p * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.child]

        '''
        w = []
        for c in self.child:
            w_c = c.q() / c.n()
            print(c.parent_action, ':', w_c)

            w.append(w_c)
        '''

        return self.child[np.argmax(w)]

    def rollout_policy(self, moves):
        move_index = np.random.randint(len(moves))
        return moves[move_index]

    def _tree_policy(self):
        current = self
        while not current.is_game_over():
            if not current.is_fully_expanded():
                return current.expand()
            else:
                current = current.best_child()
        
        return current

    def best_action(self):
        sim_no = 1000

        for i in range(sim_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child(0).parent_action

class MCTSAgent:
    def __init__(self, color):
        self.color = color

    def get_action(self, state):
        return Node(state, color=self.color).best_action()

class UserAgent:
    def __init__(self, color):
        self.color = color

    def get_action(self, state):
        print_state(state.board)
        return tuple(map(int, input().split(' ')))

def print_state(state):
    chs = {
         1: 'X',
         0: ' ',
        -1: 'O'
    }

    for i in range(5):
        if (i % 2 == 0):
            print(''.join(chs[state[i//2,j//2]] if j % 2 == 0 else '│' for j in range(5)))
        else:
            print('─┼─┼─')

state = State(np.zeros((3, 3)))
agents = {
    1: UserAgent(1),
    -1: MCTSAgent(-1)
}

turn = 1
while not state.is_game_over():
    action = agents[turn].get_action(state)
    state = state.move(action, turn)
    turn *= -1

print_state(state)