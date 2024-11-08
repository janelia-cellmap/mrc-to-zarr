import zarr 
import numpy
import mrcfile
from numcodecs import Zstd
import os
import click
from typing import Tuple 
from dask_jobqueue import LSFCluster
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client, wait, LocalCluster
from toolz import partition_all
import time

def generate_multiscales_metadata(
    ds_name: str,
    voxel_size: list,
    translation: list,
    units: list,
    axes: list)->dict:
    """Generate ome-ngff style multiscale metadata for Zarr array.
    
    Args:
        ds_name (str): name of the zarr array
        voxel_size (list): size of the voxel in units
        translation (list): shift in coordinate space
        units (str): physical units
        axes (list): axes order 

    Returns:
        dict: ome-ngff Zarr multiscale attributes 
    """
    z_attrs: dict = {"multiscales": [{}]}
    z_attrs["multiscales"][0]["axes"] = [
        {"name": axis, "type": "space", "unit": unit} for axis, unit in zip(axes, units)
    ]
    z_attrs["multiscales"][0]["coordinateTransformations"] = [
        {"scale": [1.0, 1.0, 1.0], "type": "scale"}
    ]
    z_attrs["multiscales"][0]["datasets"] = [
        {
            "coordinateTransformations": [
                {"scale": voxel_size, "type": "scale"},
                {"translation": translation, "type": "translation"},
            ],
            "path": ds_name,
        }
    ]

    z_attrs["multiscales"][0]["name"] = ""
    z_attrs["multiscales"][0]["version"] = "0.4"

    return z_attrs

def initialize_dask_client(cluster_type : str)->Client:
    """Initialize dask client.

    Args:
        cluster_type (str): type of the cluster, either local or lsf

    Returns:
        (Client): instance of a dask client
    """
    num_cores = 1
    if cluster_type=='lsf':
        cluster = LSFCluster(
            cores=num_cores,
            processes=num_cores,
            memory=f"{15 * num_cores}GB",
            ncpus=num_cores,
            mem=15 * num_cores,
            walltime="48:00",
            local_directory = "/scratch/$USER/"
            )
    elif cluster_type=='local': 
        cluster = LocalCluster()
        
    client = Client(cluster)
    with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w") as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)
    return client
     
def save_chunk(src_path : str,
                  z_arr: zarr.core.Array,
                  chunk_slice : Tuple[slice, ...]):
    """Copies data from a particular part of the input mrc array into a specific chunk of the output zarr array.

    Args:
        src_path (str): path to the input mrc file 
        z_arr (zarr.core.Array): output zarr array object
        chunk_slice (Tuple[slice, ...]): slice of the mrc array to copy. 
    """
    mrc_file = mrcfile.mmap(src_path, mode='r')

    #if not (mrc_file.data[chunk_slice] == 0).all():
    z_arr[chunk_slice] = mrc_file.data[chunk_slice]
    

def mrc_to_zarr(src_path: str,
                dest_path: str,
                client: Client,
                scale : list,
                translation: list,
                axes : list,
                units: list):
    """Use mrcfile memmap to access small parts of the mrc file and write them into zarr chunks.

    Args:
        src_path (str): path to the input mrc dataset.
        dest_path (str): path to the zarr group where the output dataset is stored.
        client (Client): instance of a dask client
        scale (list): size of the voxel in units
        translation (list): shift in coordinate space
        units (str): physical units
        axes (list): axes order 
    """
    # default compression spec 
    
    comp = Zstd(level=6)
    
    mrc_file = mrcfile.mmap(src_path, mode='r')
    
    zs = zarr.NestedDirectoryStore(dest_path)
    ds_name = 's0'
    z_root = zarr.open(zs, mode='a')
    z_arr = z_root.require_dataset(
                        name=ds_name,
                        shape=mrc_file.data.shape,
                        chunks =(128,)*mrc_file.data.ndim,
                        dtype=mrc_file.data.dtype,
                        compressor=comp)
    
    out_slices = slices_from_chunks(normalize_chunks(z_arr.chunks, shape=z_arr.shape))
    out_slices_partitioned = tuple(partition_all(100000, out_slices))

    for idx, part in enumerate(out_slices_partitioned):
        
        print(f'{idx + 1} / {len(out_slices_partitioned)}')
        start = time.time()
        fut = client.map(lambda v: save_chunk(src_path, z_arr, v), part)
        print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
        # wait for all the futures to complete
        result = wait(fut)
        print(f'Completed {len(part)} tasks in {time.time() - start}s')
                
    #generate data and save to multiscale 
    z_attrs = generate_multiscales_metadata(ds_name, scale, translation, units, axes)
    z_root.attrs['multiscales'] = z_attrs['multiscales']

    

@click.command()
@click.option('--src','-s', type=click.Path(exists = True),help='Input .mrc file location.')
@click.option('--dest', '-d', type=click.STRING,help='Output .zarr file location.')
@click.option(
    "--translation",
    "-t",
    nargs=3,
    default=(0.0, 0.0, 0.0),
    type=float,
    help="Metadata translation(offset) value. Order matters. \n Example: -t 1.0 2.0 3.0",
)
@click.option(
    "--scale",
    "-s",
    nargs=3,
    default=(1.0, 1.0, 1.0),
    type=float,
    help="Metadata scale value. Order matters. \n Example: -s 1.0 2.0 3.0",
)
@click.option(
    "--axes",
    "-a",
    nargs=3,
    default=("z", "y", "x"),
    type=str,
    help="Metadata axis names. Order matters. \n Example: -a z y x",
)
@click.option(
    "--units",
    "-u",
    nargs=3,
    default=("nanometer", "nanometer", "nanometer"),
    type=str,
    help="Metadata unit names. Order matters. \n Example: -u nanometer nanometer nanometer",
)
@click.option('--workers','-w',default=100,type=click.INT,help = "Number of dask workers")
@click.option('--cluster', '-c', default='' ,type=click.STRING, help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'")
def cli(src, dest, cluster, workers,  scale, translation, units, axes):
    
    client = initialize_dask_client(cluster)
    client.cluster.scale(workers)

    mrc_to_zarr(src, dest, client, scale, translation, axes, units)
 
    client.cluster.scale(0)

        
if __name__ == '__main__':
    cli()