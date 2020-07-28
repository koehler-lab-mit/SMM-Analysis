"""
This module contains functions for input and output of files following GenePix-compliant file formats.
File definitions can be found here: http://mdc.custhelp.com/app/answers/detail/a_id/18883/kw/18883
"""
import csv
import re
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

_notext = ' \t\n\r\x0b\x0c,"'


def _standardize_names(pd_object):
    return pd_object.str.upper().str.replace('.', '')


def load_atf(path, sep='\t', clean_names=True, quote_strings=True, **kwargs):
    """
    Read an ATF-compliant CSV.
    Headers are in the form 'key'='value'.
    Other kwargs are passed to pandas.read_csv.

    :param path: A file path
    :param quote_strings: If True, quoted fields within the ATF file will be
        preserved as strings. Otherwise, Pandas will infer the type.
    :param sep: The separator used within the data table
    :return: A tuple with fields headers and data
    """
    headers = {}
    with open(path, 'r') as file:
        file.readline()
        n_headers, n_fields = re.findall(r'(\d+)', file.readline())
        for _ in range(int(n_headers)):
            key, val = file.readline().split('=', 2)
            headers[key.strip(_notext)] = val.strip(_notext)
        if quote_strings:
            data_start = file.tell()
            field, quoted = csv.reader([file.readline(), file.readline()],
                                       delimiter=sep)
            str_fields = {x: str for x, y in zip(field, quoted) if y[0] == '"'}
            file.seek(data_start)
        else:
            str_fields = None
        data = pd.read_csv(file, sep=sep,
                           skipinitialspace=True,
                           converters=str_fields,
                           na_values=['Error'],
                           **kwargs)
        if clean_names:
            data.columns = _standardize_names(data.columns)
        data.attrs['headers'] = headers
        return data


def load_gal(path, map_blocks=False, **kwargs):
    data = load_atf(path, **kwargs)
    blocks = {}
    for key, val in data.attrs['headers'].items():
        if re.match(r"Block\d+", key):
            blocks[int(key[5:])] = val.split(",", 7)
    blocks = pd.DataFrame.from_dict(blocks, 'index').apply(pd.to_numeric)
    if not blocks.empty:
        blocks.columns = ["X", "Y", "Dia", "nX", "dX", "nY", "dY"]
        data.attrs['blocks'] = blocks
    if map_blocks and not blocks.empty:
        # nX and nY refer to spot indices in rows or columns, respectively
        # dX and dY refer to the spacings between each spot along each axis
        # The general form of these equations is (block)+(spacing)*(index)
        data.insert(3, 'X', data.BLOCK.map(blocks.X) + (data.COLUMN - 1) * data.BLOCK.map(blocks.dX))
        data.insert(4, 'Y', data.BLOCK.map(blocks.Y) + (data.ROW - 1) * data.BLOCK.map(blocks.dY))
        data.insert(5, 'DIA', data.BLOCK.map(blocks.Dia))
    return data


load_gpr = load_atf


class Scan(np.ndarray):
    """
    A microarray scan of a single channel.
    This is a numpy array with an extra attribute, 'attrs'. That's all.
    attrs is a namespace with scan attributes, e.g. channel and filter
    """

    def __array_finalize__(self, obj):
        self.attrs = getattr(obj, 'attrs', None)


def load_tif(path, cache='all'):
    """
    Loads a tiff file following GenePix format into Scan objects
    Scan objects store data in the attribute `data`
    :param path: A path-like to the tiff
    :param cache: Either a string or list of strings specifying channels to load
    into memory, the string 'all' to load all channels into memory, or None
    :return: A list of Scan objects
    """
    path = Path(path)

    if isinstance(cache, str):
        cache = [cache]
    image = Image.open(path)
    tags = image.tag_v2
    basic_info = dict(
        Make=tags.get(271, None),
        Model=tags.get(272, None),
        Software=tags.get(305, None),
        DateTime=tags.get(306, None),
        Barcode=path.stem,
        resolution=1,
        x_offset=0,
        y_offset=0,
        Filter=None
    )
    basic_info.update(x.split('=') for x in re.findall(r"\w+=[^;]*", tags.get(315, '')))

    images = []
    for n in range(image.n_frames):
        image.seek(n)
        label = image.tag_v2[270]
        if re.search('\\[W[0-9]*]', label):
            channel = re.match(r'\S+', label).group()
            if ('all' in cache) or (channel in cache):
                data = np.asarray(image)
                info = basic_info.copy()
                info.update(dict(
                    channel=channel,
                    resolution=10000 / float(tags[282]),
                    x_offset=10000 * float(tags[286]),
                    y_offset=10000 * float(tags[287])))
                info.update(x.split('=') for x in re.findall(r"\w+=[^;]*", tags.get(316, '')))
                info['Barcode'] = info.get("Barcode", path.stem)
                data = data.view(Scan)
                data.attrs = SimpleNamespace(**info)
                images.append(data)
    return images


def write_atf(path, headers, data):
    """
    Write an ATF-compliant tab-delimited text file
    :param path: A file path
    :param headers: A dict of headers to write above the data
    :param data: A DataFrame to save
    :return: None
    """
    with open(path, 'rw') as file:
        file.write("ATF\t1.0\n")
        file.write(f"{len(headers)}\t{data.shape[1]}\n")
        for key, val in headers.items():
            file.write(f'"{key}={val}"\n')
        data.to_csv(file, index=False, sep="\t", quoting=csv.QUOTE_NONNUMERIC)


def write_gal(path, headers, blocks, data):
    """
    Write a GAL file in GAL v1.0 format (i.e. ATF)
    :param path: A file path
    :param headers: A dict of headers to write above the data
    :param blocks: A DataFrame of block format
    :param data: A DataFrame of spot information
    :return: None
    """
    headers = OrderedDict(headers)
    headers['TYPE'] = 'GenePix ArrayList V1.0'
    headers.move_to_end('TYPE', last=False)
    blocks = blocks.applymap(str).set_index(blocks.columns[0]).agg(','.join)
    blocks.index = 'Block' + blocks.index
    headers.update(blocks.to_dict())
    write_atf(path, headers, data)


# TODO Support HDF5 input and output
def read_hdf5(path):
    pass


def write_hdf5(path, gprs=None, scans=None):
    pass
