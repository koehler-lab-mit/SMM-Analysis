from pathlib import Path
from itertools import chain
import re

import numpy as np
import pandas as pd
from PIL import Image


class Scan:

    def __init__(self, data, laser, filter, resolution, x_offset, y_offset, barcode='', info=None):
        self.data = np.ascontiguousarray(data)
        self.laser = laser
        self.filter = filter
        self.resolution = resolution
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.barcode = barcode
        self.info = info or {}

    @classmethod
    def load_tif(cls, path):
        image = Image.open(path)
        scan_info = dict(x.split('=') for x in re.findall(r"\w+=[^;]*", image.tag_v2.get(315, '')))
        scans = []
        for frame in range(image.n_frames):
            image.seek(frame)
            if re.search(r"\[P.*]", image.tag_v2[270]) is None:
                data = np.asarray(image)
                tags = image.tag_v2
                info = dict(x.split('=') for x in re.findall(r"\w+=[^;]*", image.tag_v2.get(316, '')))
                info.update(scan_info)
                barcode = info.get('Barcode', Path(image.filename).stem)
                filter = info.get('Filter', None)
                laser = re.match(r"\S+", tags[270]).group()
                resolution = 10000 / float(tags[282])
                offset = (float(tags.get(286, 0)), float(tags.get(287, 0)))
                scan = Scan(data, laser, filter, resolution, *offset, barcode, info)
                scans.append(scan)
        return scans

    def quantify(self, x, y, d):
        pass

    def crop_convert(self, y1, y2, x1, x2):
        y1 = (y1 - self.y_offset) / self.resolution
        y2 = (y2 - self.y_offset) / self.resolution
        x1 = (x1 - self.x_offset) / self.resolution
        x2 = (x2 - self.x_offset) / self.resolution
        try:
            return [self.data[a:b, c:d] for a,b,c,d in zip(y1,y2,x1,x2)]
        except TypeError:
            return self.data[y1:y2, x1:x2]

    def __str__(self):
        return f'Scan({self.barcode}, {self.laser}nm, {self.filter})'


def load_tifs(paths):
    return pd.DataFrame.from_records(
        ((x.barcode, x.laser, x.filter, x) for x in chain.from_iterable(map(Scan.load_tif, paths))),
        columns='BARCODE LASER FILTER SCAN'.split()
    )


# TODO Support HDF5 input and output
def read_hdf5(path):
    pass


def write_hdf5(path, gprs=None, scans=None):
    pass
