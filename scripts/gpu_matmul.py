###############################################################################
###
### Perform matrix multiplication using cupy on GPU.
###
### Written by Kelly Marchisio, Oct 2020.
###
###############################################################################

import argparse
import cupy as cp
import dask.array as da
import h5py
import math
import numpy as np
import time

GPU_IND_LIMIT=250000000
GPU_COL_LIMIT=5000
SPLIT_SIZE=500

def main():
    parser = argparse.ArgumentParser(description='Train an unsupervised SMT model')
    parser.add_argument('--m1', metavar='PATH', required=True, 
        help='First matrix to multiply.')
    parser.add_argument('--m2', metavar='PATH', required=True, 
        help='Second matrix to multiply.')
    parser.add_argument('--ofile', metavar='PATH', required=True, 
        help='Output filename for result of matrix multiplication.')
    parser.add_argument('--cpu_only', type=int, default=0,
        help='Perform operations on CPU only.')
    parser.add_argument('--fp16', type=int, default=1,
        help='Use float16')
    parser.add_argument('--hdf5', type=int, default=0,
        help='Save as HDF5')

    args = parser.parse_args()
    if args.fp16:
        dtype='float16'
    else:
        dtype='float32'

    if args.cpu_only:
        print('Loading with numpy')
        xp = np
    else:
        print('Loading with cupy')
        xp = cp

    A = xp.load(args.m1)
    B = xp.load(args.m2)
    if A.shape[1] != B.shape[0]:
        print('Error: Array dimensions do not match.', (A.shape, B.shape))
        exit(1)
    if not args.cpu_only and A.shape[0] * B.shape[1] > GPU_IND_LIMIT:
        print('Final matrix will be large!', (A.shape[0], B.shape[1]))
        print('Computing column-by-column...')
        AB = None
        prev_i = 0
        for i in range(GPU_COL_LIMIT, B.shape[1], GPU_COL_LIMIT):
            AB_js = xp.matmul(A, B[:, prev_i: i])
            AB_js_n = xp.asnumpy(AB_js)
            AB = concat(AB, AB_js_n, 1)
            prev_i = i
            print('Iteration:', i)
            print('Current shape of AB:', AB.shape)
        AB_js = xp.matmul(A, B[:, prev_i: ])
        AB_js_n = xp.asnumpy(AB_js)
        AB = concat(AB, AB_js_n, 1)
    else:
        AB = xp.dot(A, B)

    # Other methods I tried:
    print('Type of array to save:', AB.dtype)
    print('Current shape of AB:', AB.shape)
    # AB = AB.rechunk((5000,AB.shape[1]))
    if args.hdf5:
        da.to_hdf5(args.ofile + '.hdf5', '/AB', AB)
    else:
        xp.save(args.ofile, AB)
    print('Matrix multiplication done.')


def concat(AB, AB_js_n, axis):
    if AB is None:
        AB = da.from_array(AB_js_n)
    else:
        # Cite for making list of dask.arrays to append individually to not
        # make memory consumption very high:
        # https://github.com/dask/dask/issues/5601
        split = SPLIT_SIZE
        nsplits = math.floor(AB_js_n.shape[1] / split)
        sliced = [da.from_array(AB_js_n[:, ind*split: ind*split + split],
            chunks=('auto', 'auto')) for ind in range(nsplits)]
        l = [AB]
        l.extend(sliced)
        AB = da.concatenate(l, axis=axis)
        if AB_js_n.shape[1] % split > 0:
            last_slice = da.from_array(
                    AB_js_n[:, nsplits*split: nsplits*split + split],
                    chunks=('auto', 'auto'))
            AB = da.concatenate([AB,last_slice], axis=axis)
    return AB


if __name__ == '__main__':
    main()
