import chess

import csv

import numpy as np

GAMES = 'parsedData/games.txt'

def parse_moves(moves):
    """returns the vectorized data represented by moves"""
    moves = moves.split(" ")
    board = chess.Board()

    data = []
    for move in moves:
        board.push_san(move)
        data += parse_board(str(board).split())

    return data



def parse_board(board):
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

    return vector

# extend to file instead
def parse_data():
    csvfile = open('games.csv', 'r')
    filereader = csv.reader(csvfile, delimiter=',')
    next(filereader)
    # set this value for number of data points parsed -1 for all
    i = -1
    first = True

    file = open(GAMES, 'w')

    for ind, line in enumerate(filereader):
        if (ind == i):
            break

        if (ind % 1000) == 0:
            print(ind)

        label = line[6]
        if label == 'white':
            value = -1
        elif label == 'black':
            value = 1
        else:
            value = 0

        for ind, piece in enumerate(parse_moves(line[12])):
            if ind % 768 == 0:
                file.write("\n")
                file.write(f"{value},")
            file.write(str(piece))

    file.close()

if __name__ == '__main__':
    parse_data()
