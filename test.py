from typing import List


def exist(board: List[List[str]], word: str) -> bool:
    m, n = len(board), len(board[0])

    def dfs(cur_i, cur_j, cur_word, cur_set):
        if cur_word == word:
            return True

        if len(cur_word) >= len(word):
            return

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for x, y in directions:
            child_i, child_j = cur_i + x, cur_j + y
            if 0 <= child_i < m and 0 <= child_j < n:
                letter = board[child_i][child_j]
                cur_set.add((child_i, child_j))
                res = dfs(child_i, child_j, cur_word + letter, cur_set)
                cur_set.remove((child_i, child_j))

                if res:
                    return True

    for i in range(m):
        for j in range(n):
            if board[i][j] == word[0]:
                if dfs(i, j, board[i][j], {(i, j)}):
                    return True

    return False


exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABCCED")
