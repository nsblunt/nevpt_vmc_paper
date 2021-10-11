#!/usr/bin/python

import optparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def extract_energies():

    eigvs = []

    data_file = 'stoch_samples_0.dat'
    file_exists = True
    try:
        f = open(data_file)
    except IOError:
        file_exists = False

    if file_exists:
        for line in f:
            if not "#" in line:
                words = line.split()
                energy = float(words[5])
                eigvs.append(energy)
        f.close()

    return eigvs

def parse_options(args):

    parser = optparse.OptionParser(usage = __doc__)
    (options, filenames) = parser.parse_args(args)
    
    return options

if __name__ == '__main__':

    (options) = parse_options(sys.argv[1:])

    exact_eigv = -0.264848349251
    plt.axvline(exact_eigv, color='black', linewidth=2.5, linestyle='dashed')

    eigvs = []
    eigvs = extract_energies()

    min_eigv = min(eigvs)
    max_eigv = max(eigvs)
    nbins = 50
    fraction = (max_eigv-min_eigv)/nbins
    eigv_bins = []
    for i in range(0,nbins+1):
        eigv_bins.append(min_eigv+i*fraction)

    plt.hist(eigvs, bins=eigv_bins, color='blue', normed=True)

    plt.show()

