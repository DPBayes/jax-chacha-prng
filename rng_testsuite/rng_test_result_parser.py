# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University
""" Tiny script to parse outputs of TestU01 battery and return success (0) or error (1). """

if __name__ == '__main__':
    import sys
    retcode = 1
    for line in sys.stdin:
        print(line, end="")
        if 'All tests were passed' in line:
            retcode = 0
    exit(retcode)
