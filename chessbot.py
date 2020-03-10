import chess
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class ChessBot():

    def __init__(self, white):
        self.colIndex = 2 if white else 0

        self.models = []
        self.load_model()



    def load_model(self):
        for file in os.listdir("."):
            if file.endswith(".sav"):
                file = open(file, 'rb')
                self.models.append(pickle.load(file))
        return self.models


    def predict_best_moves(self, board):
        possible_moves = board.legal_moves
        moves_vector = []
        moves = []
        for move in possible_moves:
            board.push(move)
            moves_vector.append(self.parse_board(str(board).split()))
            moves.append(str(board.pop()))

        moves_vector = np.concatenate(moves_vector, axis=0)
        # print(len(moves_vector[0]))
        probs = self.predict_winning_possiblities(moves_vector)

        changes = [(probs[i][self.colIndex], moves[i]) for i in range(len(moves))]
        return sorted(changes, key=lambda x: 1 - x[0])

    def predict_winning_possiblities(self, vector):
        probs = None
        for model in self.models:
            predictions = model.predict_proba(vector)
            if probs is None:
                probs = predictions
            else:
                probs += predictions

        return probs / len(self.models)

    def parse_board(self, board):
        """ the return will be a vector of length 768
        every 12 elements represents what items are stored in the block
        going from left to right top to bottom.
        the 12 element indicies are as described
        0 = white pawn 1 = white rook 2 = white knight 3 = white bishop
        4 = white queen 5 = white king
        and black pieces go in that same cycle
        """
        vector = []
        convert = {'p': 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5}
        for piece in board:

            block = [0 for i in range(12)]

            scale = 0 if piece.isupper() else 6
            if piece != '.':
                block[scale + convert[piece.lower()]] = 1

            vector += block

        vector = np.array(vector)
        vector = vector.reshape(1, len(vector))
        return vector

if __name__ == '__main__':
    bot = ChessBot(True)
    board = chess.Board()
    board.push_san('e4')
    board.push_san('e5')
    for move in bot.predict_best_moves(board):
        print(move)
