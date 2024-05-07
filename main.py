import os
import argparse
from structure import structure_propagation

parser = argparse.ArgumentParser(description='set option')
 
parser.add_argument('-i', "--image_name", default="pumpkin", help='Image Name')
parser.add_argument('-s', "--sampling_interval", type = int, default=15, help="Sampling Interval")
parser.add_argument('-p', "--patch_size", type=int, default=15, help="Patch Size")
parser.add_argument('-d', "--down_sample", type=int, default=2, help="Down Sample")
parser.add_argument('-t', "--to_show", action="store_true", help="show detail by plt", default=False)

args = parser.parse_args()
structure_propagation(args.image_name, args.sampling_interval, args.patch_size, args.down_sample, args.to_show)

