class game():
    """A simple 3x3 Tic Tac Toe board.

    Cells are stored as integers and take one of three values:
        `empty`  (-1) - the cell has not been played
        `naught` ( 0) - played by the "O" player
        `cross`  ( 1) - played by the "X" player
    """

    def __init__(self):
        """Create a fresh game with an empty 3x3 board."""
        # Use a list comprehension so each row is a distinct list object.
        # Multiplying a nested list (e.g. `[[-1]*3]*3`) would alias the same
        # row three times, and a move in one row would mutate all of them.
        self.board = [[-1, -1, -1] for _ in range(3)]
        self.naught = 0
        self.cross = 1
        self.empty = -1

    def move(self, i, j, value):
        """Place `value` at row `i`, column `j`.

        Parameters
        ----------
        i, j : int
            Row and column indices in the range [0, 2].
        value : int
            The player token to place (`self.naught` or `self.cross`).

        Returns
        -------
        bool
            True if the move was applied, False if the coordinates were out
            of bounds or the target cell was already occupied.
        """
        # Guard against bad indices before touching the grid, otherwise a
        # negative index would silently wrap around to the other side.
        if not (0 <= i < 3 and 0 <= j < 3):
            print(f"Invalid Move Operation {i},{j},{value}: Out of bounds.")
            return False

        # Reject writes to a cell that is already taken.
        if self.board[i][j] != self.empty:
            print(f"Invalid Move Operation {i},{j},{value}: You cannot place there.")
            return False

        # Note: the original draft used `==` here, which is a comparison and
        # would not actually place the token. Assignment requires `=`.
        self.board[i][j] = value
        return True

    def check_win(self):
        """Return the winning player's value, or `self.empty` if no winner.

        A win is three matching, non-empty tokens along any row, column, or
        diagonal. This does NOT detect a draw - callers should combine this
        with an "all cells filled" check to decide if the game is over.
        """
        b = self.board

        # Rows and columns share the loop index `i`, so they are checked in
        # the same pass. The `!= self.empty` guard prevents a row of three
        # empty cells from being reported as a win.
        for i in range(3):
            # Row i: b[i][0], b[i][1], b[i][2]
            if b[i][0] != self.empty and b[i][0] == b[i][1] == b[i][2]:
                return b[i][0]
            # Column i: b[0][i], b[1][i], b[2][i]
            if b[0][i] != self.empty and b[0][i] == b[1][i] == b[2][i]:
                return b[0][i]

        # Main diagonal: top-left to bottom-right.
        if b[0][0] != self.empty and b[0][0] == b[1][1] == b[2][2]:
            return b[0][0]

        # Anti-diagonal: top-right to bottom-left.
        if b[0][2] != self.empty and b[0][2] == b[1][1] == b[2][0]:
            return b[0][2]

        # No alignment found - the game is either ongoing or a draw.
        return self.empty

    def check_draw(self):
        """Return True if the game is a draw (board full, no winner).

        A draw requires two conditions to hold at the same time:
            1. No player has three in a row (`check_win` returns `empty`).
            2. Every cell on the board has been played.

        Returning False does NOT mean someone has won - it just means the
        game is not a draw. Use `check_win` to distinguish "ongoing" from
        "someone won".
        """
        # If there is a winner, the game is decided - not a draw.
        if self.check_win() != self.empty:
            return False

        # Flatten the grid and confirm no cell is still empty. `all` short
        # circuits, so it stops at the first empty cell it finds.
        return all(cell != self.empty for row in self.board for cell in row)
