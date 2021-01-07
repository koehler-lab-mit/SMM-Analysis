from base64 import b64encode
from functools import wraps
from smm_io import BytesIO
import warnings
import re

import numpy as np
import pandas as pd
import qgrid
from PIL import Image
from ipywidgets import HTML
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem.Draw import rdMolDraw2D

rdBase.DisableLog('rdApp.error')
pd.set_option('display.max_colwidth', None)

image_size = (150, 150)
image_scale = 2

def _lut(bright):
    return np.linspace(0, bright, 256, dtype=np.uint8)


_colors = {'Standard Red': _lut([255, 0, 0]),
           'Standard Yellow': _lut([255, 128, 0]),
           'Standard Green': _lut([0, 255, 0]),
           'Standard Blue': _lut([0, 75, 255]),
           None: _lut([255, 255, 255])}


def add_color(channel, color):
    _colors[channel] = _lut(color)


def get_color(channel):
    if channel in _colors:
        return _colors[channel]
    warnings.warn("This channel does not have an assigned color")
    return _colors[None]


def draw_molecule(smiles, mol_size=(150, 150)):
    drawer = rdMolDraw2D.MolDraw2DSVG(*mol_size)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = rdMolDraw2D.PrepareMolForDrawing(mol)
            drawer.DrawMolecule(mol)
    except TypeError:
        pass
    drawer.FinishDrawing()
    svg = re.search(r"<svg.*</svg>", drawer.GetDrawingText(), re.DOTALL).group()
    return svg


def draw_molecules(smiles, mol_size=(200, 200)):
    return ('<div style = "display:flex;">\n' +
                '\n'.join(draw_molecule(x, mol_size) for x in smiles) +
                '\n</div>')


def _pil_to_html(image):
    buffered = BytesIO()
    image.save(buffered, format="BMP")
    buffered = b64encode(buffered.getvalue()).decode()
    return f'<img max-width=100% style="image-rendering: pixelated;" src="data:image/bmp;base64,{buffered}">'


def _stack_images(images):
    return ('<div style = "display:flex;">\n' +
            '\n'.join(map(_pil_to_html, images)) +
            '\n</div')


def _crop_scan(spot):
    scan = spot.SCAN
    resolution = scan.resolution
    r = (spot.DIA * image_scale) / (2 * resolution)
    yp = (spot.Y - scan.offset[1]) / resolution
    xp = (spot.X - scan.offset[0]) / resolution
    return spot.SCAN.as_numpy[int(yp - r):int(yp + r), int(xp - r):int(xp + r)]


def _find_clip(images, lower, upper):
    clips = np.vstack(images.map(lambda x: np.percentile(x, (lower, upper))))
    lower = np.min(clips[:, 0])
    upper = np.max(clips[:, 1])
    return [[lower, upper]]


def _rescale_image(image, contrast):
    low, high = contrast
    np.clip(image, low, high, out=image)
    return ((image - low) * 255 / float(high - low)).astype(np.uint8)


def _numpy_to_pil(scan):
    image = Image.fromarray(scan, 'P').resize(image_size, resample=0)
    if scan.filter in _colors:
        image.putpalette(_colors[scan.filter])
    return image


class SpotTable(HTML):
    def __init__(self, rows=None, cols='LASER', contrast=(20, 98), contrast_groups=('LASER', 'FILTER')):
        super().__init__()
        self.contrast = contrast
        self.contrast_groups = contrast_groups
        self.rows = rows
        self.cols = cols

    def show_table(self, df):
        if 'LASER' not in df:
            df['LASER'] = [x.laser for x in df.SCAN]
        if 'FILTER' not in df:
            df['FILTER'] = [x.filter for x in df.SCAN]
        contrast_groups = [df[x] for x in self.contrast_groups]

        images = df[['SCAN', 'X', 'Y', 'DIA']].apply(
            lambda x: np.sqrt(_crop_scan(x)), axis=1, result_type='reduce')

        percentiles = images.groupby(contrast_groups).transform(_find_clip, 20, 98)
        images = list(map(_rescale_image, images, percentiles))
        df['IMAGE'] = map(_numpy_to_pil, images)

        df = df.reset_index(drop=True)
        df = df.pivot_table('IMAGE', self.rows, self.cols, _stack_images,
                            fill_value=Image.new('RGB', (1, 1), (255, 255, 255, 0)))
        df = df.applymap(_pil_to_html)
        self.value = df.to_html(escape=False, index_names=False,
                                index=(self.rows is not None),
                                header=(self.cols is not None))


class TableDisplay(qgrid.QGridWidget):

    @classmethod
    def show_grid(cls, *args, **kwargs):
        x = qgrid.show_grid(*args, **kwargs)
        x.__class__ = cls
        return x

    def structure(self, output=None, lookup=None):
        if output is None:
            output = HTML()

        def callback(_, df):
            df = df.get_selected_df()
            if lookup:
                smiles = lookup(df['ID'])
            else:
                smiles = df['SMILES']
            output.value = draw_molecules(smiles)

        self.on('selection_changed', callback)
        return output

    def spots(self, contrast=(20, 98),
              rows=None, cols=None, through=None, scans=None,
              **kwargs):

        spots = SpotTable()

        def callback(_, df):
            df = df.get_selected_df()
            if through is not None:
                df = pd.merge(df, through)
            if scans is not None:
                df = pd.merge(df, scans, on='BARCODE')
            # TODO Apply filters from kwargs
            spots.show_table(df)

        self.on('selection_changed', callback)
        return spots

    @wraps(pd.DataFrame.to_csv)
    def to_csv(self, *args, **kwargs):
        return self.qgrid.get_changed_df().to_csv(*args, **kwargs)


qgrid_options = {
    # SlickGrid options
    'forceFitColumns': False,  ##
    'defaultColumnWidth': 80,  ##
    'enableColumnReorder': True,  ##
    'autoEdit': False,  # ??

    # Qgrid options
    'maxVisibleRows': 6,  ##
    'minVisibleRows': 1,  ##
    'filterable': True,  ##
    'highlightSelectedCell': True,  ##
}

qgrid_col_options = {
    # SlickGrid column options
    'defaultSortAsc': True,
    'maxWidth': None,
    'minWidth': 30,
    'resizable': True,
    'sortable': True,
    'toolTip': "",
    'width': None,

    # Qgrid column options
    'editable': True
}


def show_spot_table(spots, slides, rows, cols, scale_by, contrast=(20, 98)):
    df = spots.join(slides, 'BARCODE')
    contrast_groups = [df[x] for x in scale_by]
    images = df[['SCAN', 'X', 'Y', 'DIA']].apply(
        lambda x: np.sqrt(_crop_scan(x)), axis=1, result_type='reduce')

    percentiles = images.groupby(contrast_groups).transform(_find_clip, *contrast)
    images = map(_rescale_image, images, percentiles)
    df['IMAGE'] = map(_numpy_to_pil, images)
    df = df.groupby(rows+cols)['IMAGE'].aggregate(_stack_images)
    if cols:
        df = df.unstack(cols)
    return df.to_html(escape=False, index_names=False,
                      index=(rows is not None),
                      header=(cols is not None))
