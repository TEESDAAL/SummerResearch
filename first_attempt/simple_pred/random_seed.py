import sys

def seed() -> int:
    """Return the random seed to use, returns either the first keyword argument, or 0 if none are provided."""
    try:
        return int(sys.argv[1])
    except IndexError:
        return 0
