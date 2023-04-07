#!/usr/bin/env python3

"""CSI logs captured from serial port sometimes end in an invalid byte instead of newline character. This throws an exception when read with `pd.read_csv`. 
This script strips away the character from csi logs.

    Example usage: ./logclean.py dataset/raw/jump-3.csi 
"""

import sys

file = sys.argv[1]

with open(file) as fp:
    lines = fp.readlines()
    cleaned = [line[:-1] for line in lines]

with open(file + ".cleaned", "w") as fp:
    print(*cleaned, sep="\n", file=fp)
