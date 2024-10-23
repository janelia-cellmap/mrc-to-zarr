import zarr 
import numpy
import mrcfile
from numcodecs import Zstd
import os
import click
from typing import Tuple 


def slice_along_z_dim(arr_shape : Tuple, step : int) -> list:
    """Get a list of slices of the mrc dataset that are being used to convert mrc dataset to zarr dataset. Slicing is necessary, since oftentimes whole dataset is larger than the RAM size.

    Args:
        arr_shape (Tuple): shape of the mrc array
        step (int): slicing occurs along z-direction, it is reasonable to choose minimal slicing step equal to the size of the zarr chunk in z-direction

    Returns:
        list: list of slices along z-dimension of the mrc array that are being copied one-by-one
    """
    z_len = arr_shape[0]
    slices = []
    for sl in range(0, z_len, step):
        if sl + step < z_len: 
            slices.append(slice(sl, sl+step))
        else:
            slices.append(slice(sl, z_len))
    return slices


def store_mrc_to_zarr(src_path: str,
                      dest_path: str):
    """Use mrcfile memmap to access small parts of the mrc file and write them into zarr chunks.

    Args:
        src_path (str): path to the input mrc dataset
        dest_path (str): path to the zarr group where the output dataset is stored. 
    """
    # default compression spec 
    comp = Zstd(level=6)
    
    large_mrc = mrcfile.mmap(src_path, mode='r')
    
    zs = zarr.NestedDirectoryStore(dest_path)
    z_arr = zarr.require_dataset(store=zs, 
                        path='s0',
                        shape=large_mrc.data.shape,
                        chunks =(128,)*large_mrc.data.ndim,
                        dtype=large_mrc.data.dtype,
                        compressor=comp)
    
    slices = slice_along_z_dim(large_mrc.data.shape, z_arr.chunks[0])
    for sl in slices:
        z_arr[sl] = large_mrc.data[sl]
        print(large_mrc.data.shape[0], sl)
 
 
@click.command()
@click.option('--src','-s', type=click.Path(exists = True),help='Input .mrc file location.')
@click.option('--dest', '-d', type=click.Path(exists = True),help='Output .zarr file location.')
#@click.option('--workers','-w',default=100,type=click.INT,help = "Number of dask workers")
#@click.option('--cluster', '-c', default='' ,type=click.STRING, help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'")
def cli(src, dest):
    store_mrc_to_zarr(src, dest)
 
        
if __name__ == '__main__':
    cli()