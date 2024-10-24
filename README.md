To convert mrc file to zarr array, first install poetry project and python dependencies:

    cd PATH_TO_POETRY_PROJECT_DIRECTORY/
    poetry install


run script using cli:

    poetry run python3 src/zarr_to_mrc.py  --src=PATH_TO_MRC_FILE --dest=OUTPUT_ZARR_FILE --scale 10.0 10.0 10.0

