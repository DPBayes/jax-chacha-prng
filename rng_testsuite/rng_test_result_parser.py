if __name__ == '__main__':
    import sys
    retcode = 1
    for line in sys.stdin:
        print(line, end="")
        if 'All tests were passed' in line:
            retcode = 0
    exit(retcode)
