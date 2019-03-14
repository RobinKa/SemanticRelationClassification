from collections import OrderedDict
from glob import glob
from itertools import islice
from os import remove
from os.path import join
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_corpus(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as corpus_file:
        return corpus_file.read().split()


def get_corpora_in_path(path, ext="txt"):
    return glob(join(path, "*." + ext))


def _get_max_key_size(dictionary):
    """Returns the biggest utf-8 byte count of all keys in the dictionary"""
    # TODO: Currently added 30 because it seems that hdf5 uses more to store
    # the keys
    return 30 + (max([len(str(key[0]).encode("utf-8")) for key in dictionary.keys()]) +
                 max([len(str(key[1]).encode("utf-8")) for key in dictionary.keys()]))


def data_frame_from_dict(dictionary):
    return pd.DataFrame.from_dict(dictionary, orient="index")


def write_csv_from_dict(dictionary, path):
    """Converts `dictionary` into a `DataFrame` and saves them to `path`
    in csv format."""
    data_frame = data_frame_from_dict(dictionary)

    data_frame.to_csv(path, encoding="utf-8", float_format="%.4f")


def write_hdf_from_dict(dictionary, path, key, compression=None, compression_level=0):
    """Converts `dictionary` into a `DataFrame` and saves them to `path`
    in hdf5 format."""
    data_frame = data_frame_from_dict(dictionary)

    data_frame.to_hdf(path, key, mode="w", complib=compression,
                      complevel=compression_level)


def write_hdf_chunked_from_dict(dictionary, path, key, chunks=10000, compression=None,
                                compression_level=0):
    """Same as `write_hdf_from_dict`, but writes the dictionary in chunks for less memory usage."""
    try:
        remove(path)
    except OSError:
        pass

    # Iterate over the dictionary items in chunks until none are left
    # Append the chunks to the hdf5 file
    # NOTE: Important to use an OrderedDict for the slice so order is preserved

    min_itemsize = {"index": _get_max_key_size(dictionary)}

    dict_it = iter(dictionary.items())

    while True:
        kvs = tuple(islice(dict_it, chunks))

        if not kvs:
            break

        slice_dict = OrderedDict()
        for dict_key, dict_value in kvs:
            slice_dict[str(dict_key)] = dict_value.astype(np.float32)

        data_frame = data_frame_from_dict(slice_dict)

        data_frame.to_hdf(path, key, min_itemsize=min_itemsize, complib=compression,
                          complevel=compression_level, format="table", append=True)
