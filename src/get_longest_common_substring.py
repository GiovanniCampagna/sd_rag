from typing import Optional

def get_longest_common_substring(
    a: str, 
    b: str, 
    min_len_chars: Optional[int] = None,
) -> str:

    n, m = len(a), len(b)
    # dp[i][j] will store the length of the longest common substring ending at a[i-1] and b[j-1].
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    max_len = 0
    end_index = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            # If a[i-1] == b[j-1], the longest common substring ending at (i,j) is 1 longer than the longest substring ending at (i-1, j-1).
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_index = i

                    # Early return in case minimum overlap length is already found
                    if min_len_chars and max_len >= min_len_chars:
                        return a[end_index - max_len:end_index]

    return a[end_index - max_len:end_index]
