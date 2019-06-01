"""
This module contains files for input and output of files following GenePix-compliant file formats.
File definitions can be found here: http://mdc.custhelp.com/app/answers/detail/a_id/18883/kw/18883
"""

import re
import time
from collections import OrderedDict, namedtuple

import pandas as pd

_GAL = namedtuple("GAL", "headers blocks spots")
_GPR = namedtuple("GPR", "headers spots")


def load_gal(path):
    """
    Reads a GAL file into a Pandas DataFrame, and additionally computes the X, Y, and Radius of each spot
    :param path: A string or Path object toward the GAL file
    :return: A NamedTuple with fields headers (dict), blocks (DataFrame), and spots (DataFrame)
    """
    # TODO Added ID columns may be incorrectly inferred to be numeric. Can this be avoided?
    headers = OrderedDict()
    blocks = OrderedDict()
    with open(path, 'r') as gal:
        gal.readline()
        header_rows = int(re.match(r"\d+", gal.readline()).group()) # Indicates the number of optional header rows
        for _ in range(header_rows):
            key, val = gal.readline().strip().split('=', 2)
            if re.match(r"Block\d+", key):  # If the field describes a block, add it to the blocks dict
                blocks[int(key[5:])] = val.split(",", 7)
            else:  # If the field is just a header, add it to the header dict
                headers[key] = val
        blocks = pd.DataFrame.from_dict(blocks, 'index',columns=["X", "Y", "Dia", "nX", "dX", "nY", "dY"])
        blocks = pd.to_numeric(blocks)  # Load blocks into a DataFrame, and cast it all to numbers
        spots = pd.read_csv(gal, sep="\t", na_values="-")  # Read the rest of the file into a DataFrame

        # nX and nY refer to spot indices in rows or columns, respectively
        # dX and dY refer to the spacings between each spot along each axis
        # The general form of these equations is (Block edge)+(Spot spacing)*(Spot Index)
        spots.insert(3, 'X', spots.Block.map(blocks.X) + (spots.Column - 1) * spots.Block.map(blocks.dX))
        spots.insert(4, 'Y', spots.Block.map(blocks.Y) + (spots.Row - 1) * spots.Block.map(blocks.dY))
        spots.insert(5, 'Radius', spots.Block.map(blocks.Dia) / 2)
        spots.rename(lambda x: str.title(x).strip('.'), axis=1, inplace=True)  # Normalize the titles to Title Case
        spots.rename({"Id": "ID"}, axis=1, inplace=True)  # Except for ID, because that looks funny
        return _GAL(headers, blocks, spots)  # Can be accessed as gal.headers, gal.blocks, or gal.spots


def load_gpr(path):
    """
    Reads a GPR file into a Pandas DataFrame
    :param path: A string or Path object toward the GPR file
    :return: A NamedTuple with fields headers (dict) and spots (DataFrame)
    """
    headers = None
    spots = None
    # TODO Write the function to read GPR files.
    # Here, we assume that any added identifier fields have already been quoted. It's too complicated otherwise.
    return _GPR(headers, spots)


def write_gal(header, blocks, data, path):
    # TODO Clean up this function; all files must be tested against the GenePix Scanner for compatibility
    try:
        del header["Type"]
    except KeyError:
        pass
    columns = len(data.columns)

    def _format(string, n=columns-1, quote=True):
        if quote:
            return '"%s"%s\n' % (string, "\t"*n)
        else:
            return '%s%s\n' % (string, "\t"*n)

    optional_rows = len(header) + len(blocks) + 2
    with open(path, mode='w') as file:
        file.write(_format("ATF\t1", n=columns-2, quote=False))
        file.write(_format("%s\t%s" % (optional_rows, columns), n=columns-2, quote=False))
        file.write(_format("Type=GenePix ArrayList V1.0"))
        file.writelines(_format("%s=%s" % (k, v)) for k, v in header.items())
        file.write(_format(time.strftime("FileCreated=%x %X")))
        for line in blocks.astype(str).itertuples():
            file.write('"Block%s=%s"%s\n' % (line.Index, ','.join(line[1:]), "\t"*(columns-1)))
        data.to_csv(file, index=False, sep="\t", na_rep="-")
