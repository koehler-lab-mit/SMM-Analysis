import numpy as np
import pandas as pd
from io import BytesIO
from collections import OrderedDict
import re
import time


def load_gal(path, infer_spots=True):
    info = OrderedDict()
    blocks = {}
    block_titles = ["Block", "Column", "Row", "Name", "ID"]
    with open(path, 'r') as file:
        while True:
            index = file.tell()
            line = file.readline().strip(",\'\"\t\n,")
            if file.tell() == index: raise ValueError("Could not parse the input file")
            x = re.match("Block(?P<block>\d+)=(?P<info>.+)", line, re.I)
            if x:
                blocks[int(x.group("block"))] = re.split('[,\t]', x.group("info"))
            elif re.match(".+=.+", line):
                key, val = line.split("=", 2)
                info[key] = val.strip()
            elif all(re.search(word, line, re.I) for word in block_titles):
                file.seek(index)
                break
        blocks = pd.DataFrame.from_dict(blocks, 'index').apply(pd.to_numeric).dropna(1)
        blocks.columns = ['X', 'Y', 'Dia', 'nX', 'dX', 'nY', 'dY']
        gal = pd.read_csv(file, sep=None, engine="python", na_values="-")
        if infer_spots:
            gal.insert(3, 'X', gal.Block.map(blocks.X) + (gal.Column - 1) * gal.Block.map(blocks.dX))
            gal.insert(4, 'Y', gal.Block.map(blocks.Y) + (gal.Row - 1) * gal.Block.map(blocks.dY))
            gal.insert(5, 'Radius', gal.Block.map(blocks.Dia) / 2)
        gal.rename(lambda x: str.title(x).strip('.'), axis=1, inplace=True)
        gal.rename({"Id": "ID"}, axis=1, inplace=True)
        return gal, blocks, info


def load_gpr(path, id_columns=["Name", "ID"]):
    id_columns = list(set(id_columns) | {"Name", "ID"})
    header_info = OrderedDict()
    with open(path, 'r') as file:
        while True:
            index = file.tell()
            line = file.readline()
            if file.tell() == index: raise ValueError("Could not parse the input file")
            if re.match(".+=.+", line):
                key, val = line.split("=", 2)
                header_info[key] = val.strip()
            elif re.search("Block", line, re.I) and re.search("Column", line, re.I) and re.search("Row", line, re.I):
                file.seek(index)
                break
        gpr = pd.read_csv(file, sep=None, engine='python', dtype={x: str for x in id_columns}, na_filter=False)
        gpr[id_columns] = gpr[id_columns].replace("-", "")
    return gpr, header_info
