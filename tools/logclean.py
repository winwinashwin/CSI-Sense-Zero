#!/usr/bin/env python3

"""CSI logs captured from serial port sometimes end in an invalid byte instead of newline character. This throws an exception when read with `pd.read_csv`. 
This script strips away the character from csi logs.

    Example usage: ./logclean.py dataset/raw/jump-3.csi 
"""

import sys

input_file = sys.argv[1]
output_file = f"{input_file}.cleaned"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        f_out.write(line.rstrip() + "\n")
