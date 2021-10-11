#!/usr/bin/python

import argparse
import os
from subprocess import call

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dir_1')
    parser.add_argument('dir_2')
    parser.add_argument('--dir_3')
    args = parser.parse_args()

    print(args)

    for data_file in os.listdir(args.dir_1):
        if data_file.startswith('pt2_energies') and not 'avg' in data_file:
            if args.dir_3==None:
                script = 'cat ' + args.dir_1 + './' + data_file + ' ' + \
                                  args.dir_2 + './' + data_file + ' > ' + \
                                  './' + data_file
            else:
                script = 'cat ' + args.dir_1 + './' + data_file + ' ' + \
                                  args.dir_2 + './' + data_file + ' ' + \
                                  args.dir_3 + './' + data_file + ' > ' + \
                                  './' + data_file
            print(script)
            call(script, shell=True)
