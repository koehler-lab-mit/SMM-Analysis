from base64 import b64encode
from functools import wraps
from io import BytesIO

import numpy as np
import pandas as pd
import qgrid
from PIL import Image, ImageOps
from ipywidgets import HTML
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem.Draw import rdMolDraw2D

rdBase.DisableLog('rdApp.error')
pd.set_option('display.max_colwidth', None)

_colors = {'635': [255, 0, 0],
           '594': [255, 128, 0],
           '532': [0, 255, 0],
           '488': [0, 75, 255]}


def _join_images(images, spacing=5):
    images = [i for i in images if isinstance(i, Image.Image)]
    if len(images) == 0:
        return Image.new('RGB', (1, 1), (255, 255, 255, 0))
    elif len(images) == 1:
        return list(images)[0]
    else:
        total_width = sum(i.width for i in images) + spacing * (len(images) - 1)
        max_height = max(i.height for i in images)
        image = Image.new('RGB', (total_width, max_height), (255, 255, 255, 0))
        x = 0
        for im in images:
            image.paste(im, (x, 0))
            x += im.width + spacing
    return image


def _pil_to_html(image):
    try:
        buffered = BytesIO()
        image.save(buffered, format="BMP")
        buffered = b64encode(buffered.getvalue()).decode()
    except (AttributeError, TypeError):
        return ''
    #    return f'<img style="image-rendering: pixelated;image-rendering: -moz-crisp-edges;image-rendering: crisp-edges;" src="data:image/bmp;base64,{buffered}">'
    return f'<img width={image.width} height={image.height} src="data:image/bmp;base64,{buffered}">'


class ChemicalSVG(HTML):
    # TODO Placeholder sizing isn't exactly right; there's still some jumping
    def __init__(self, size, mol=None):
        super().__init__()
        self.default_value = '<div style="width:%spx;height:%spx;"/div>' % (size[0], size[1])
        self.value = self.default_value
        self.size = size
        if mol:
            self.show_molecule(mol)

    def show_molecule(self, mol):
        drawer = rdMolDraw2D.MolDraw2DSVG(*self.size)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        self.value = drawer.GetDrawingText()

    def clear_molecule(self):
        self.value = self.default_value


class SpotTable(HTML):
    def __init__(self, zoom=2, size=(150, 150), contrast=(20, 98),
                 rows=None, cols=None):
        super().__init__()
        self.zoom = zoom
        self.size = size
        self.contrast = contrast
        self.rows = rows
        self.cols = cols
        if (rows is None) and (cols is None):
            self.cols = 'CHANNEL'

    def show_table(self, df):
        r = (df['DIA'] * self.zoom) / (2 * df.RESOLUTION)
        yp = (df.Y - df.Y_OFFSET) / df.RESOLUTION
        xp = (df.X - df.X_OFFSET) / df.RESOLUTION
        y0 = (yp - r).astype(np.uint64)
        y1 = (yp + r).astype(np.uint64)
        x0 = (xp - r).astype(np.uint64)
        x1 = (xp + r).astype(np.uint64)
        df['IMAGE'] = list(map(self._collect_image, y0, y1, x0, x1, df.CHANNEL, df.SCAN))
        df = df.reset_index(drop=True)
        df = df.pivot_table('IMAGE', self.rows, self.cols, _join_images,
                            fill_value=Image.new('RGB', (1, 1), (255, 255, 255, 0)))
        df = df.applymap(_pil_to_html)
        self.value = df.to_html(escape=False, index_names=False,
                                index=(self.rows is not None),
                                header=(self.cols is not None))

    def _collect_image(self, y0, y1, x0, x1, channel, scan):
        image = np.sqrt(scan.data[y0:y1, x0:x1])
        if self.contrast:
            low, high = np.percentile(image, self.contrast)
            np.clip(image, low, high, out=image)
            image = (image - low) * 255 / float(high - low)
        image = image.astype(np.uint8)
        image = (Image.fromarray(image).convert(mode='L').resize(self.size))
        return ImageOps.colorize(image, [0, 0, 0], _colors[channel])


class TableDisplay(qgrid.QGridWidget):

    @classmethod
    def show_grid(cls, *args, **kwargs):
        x = qgrid.show_grid(*args, **kwargs)
        x.__class__ = cls
        return x

    def structure(self, size=(175, 175)):
        structure = ChemicalSVG(size)

        def callback(_, df):
            df = df.get_selected_df()
            if len(df.SMILES.unique())!=1:
                structure.clear_molecule()
            else:
                try:
                    smiles = df['SMILES'].pop(0)
                    mol = Chem.MolFromSmiles(smiles)
                    structure.show_molecule(mol)
                except (TypeError, KeyError):
                    structure.clear_molecule()

        self.qgrid.on('selection_changed', callback)
        return structure

    def spots(self, zoom=2, size=(150, 150), contrast=(20, 98),
              rows=None, cols=None, through=None, scans=None,
              **kwargs):

        spots = SpotTable(zoom, size, contrast, rows, cols)

        def callback(_, df):
            df = df.get_selected_df()
            if through is not None:
                df = pd.merge(df, through)
            if scans is not None:
                df = pd.merge(df, scans, on='BARCODE')
            # TODO Apply filters from kwargs
            spots.show_table(df)

        self.qgrid.on('selection_changed', callback)
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
