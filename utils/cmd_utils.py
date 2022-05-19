"""
 @author   Maksim Penkin
"""


import argparse


def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError('%s is an invalid positive int value' % value)
    return ivalue


def check_positive_float(value):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError('%s is an invalid positive float value' % value)
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser(description='Scale-Estimation arguments', usage='%(prog)s [-h]')

    parser.add_argument('--image', type=str,
                        help='path to a processing image', metavar='')
    parser.add_argument('--cnn', type=str, default='VGG19', choices=['VGG19'],
                        help='CNN to be used in scale-estimation pipeline', metavar='')
    parser.add_argument('--block2analyze', type=str, default='5', choices=['1', '2', '3', '4', '5'],
                        help='CNN`s block to be analyzed in scale-estimation pipeline', metavar='')

    args = parser.parse_args()
    return args
