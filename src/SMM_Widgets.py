from base64 import b64encode
from functools import wraps
from io import BytesIO
from operator import itemgetter

import numpy as np
import pandas as pd
import qgrid
from PIL import Image, ImageOps
from ipywidgets import HTML
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

pd.set_option('display.max_colwidth', -1)

colors = {'635': [255, 0, 0],
          '594': [255, 128, 0],
          '532': [0, 255, 0],
          '488': [0, 75, 255]}


def _join_images(images, spacing=5):
    images = [i for i in images if isinstance(i, Image.Image)]
    if len(images) == 0:
        image = Image.new('RGB', (1, 1), (255, 255, 255, 0))
    elif len(images) == 1:
        image = list(images)[0]
    else:
        widths = [i.width for i in images]
        heights = [i.height for i in images]
        size = (sum(widths) + spacing * (len(images) - 1), max(heights))
        image = Image.new('RGB', size, (255, 255, 255, 0))
        x = 0
        for im in images:
            image.paste(im, (x, 0))
            x += im.width + spacing
    return image


def _pil_to_html(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered = b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{buffered}">'


class _ImageFactory:
    def __init__(self, df_row, size, contrast):
        self.df_row = df_row[['y0', 'y1', 'x0', 'x1', 'SCAN']].copy()
        self.size = size
        self.contrast = contrast

    def __getitem__(self, channel):
        row = self.df_row
        if channel not in row.SCAN:
            return Image.new('RGB', (1, 1), (255, 255, 255, 0))
        scan = row.SCAN[channel]

        # Crop Scan
        y0 = int((row.y0 - scan.y_offset) / scan.resolution)
        y1 = int((row.y1 - scan.y_offset) / scan.resolution)
        x0 = int((row.x0 - scan.x_offset) / scan.resolution)
        x1 = int((row.x1 - scan.x_offset) / scan.resolution)
        image = scan.data[y0:y1, x0:x1]

        # Downscale to uint8
        image = np.sqrt(image)
        if self.contrast:
            low, high = np.percentile(image, self.contrast)
            np.clip(image, low, high, out=image)
            image = (image - low) * 255 / float(high - low)
        image = image.astype(np.uint8)

        # Make into PIL Image
        image = (Image.fromarray(image).convert(mode='L').resize(self.size))
        return ImageOps.colorize(image, [0, 0, 0], colors[scan.channel])


class TableDisplay:
    def __init__(self, data):
        self.data = qgrid.show_grid(
            data, show_toolbar=False,
            grid_options=qgrid_options,
            column_options=qgrid_col_options)

    def spots(self, lookup=None,
              channels=None, zoom=2, size=(150, 150), contrast=(20, 98),
              rows='ID', cols='CHANNEL'):
        spots = HTML()

        def callback(_, df):
            df = df.get_selected_df()
            if lookup:
                df = lookup(df)
            if not pd.Series(['X', 'Y', 'DIA', 'SCAN']).isin(df.columns).all():
                raise ValueError("DataFrame does not have spot-wise  information")
            r = df.DIA * zoom / 2
            df = df.assign(y0=df.Y - r, y1=df.Y + r, x0=df.X - r, x1=df.X + r)
            ims = df.apply(_ImageFactory, axis=1, args=(size, contrast))
            if channels is None:
                sub_channels = set().union(*(x.keys() for x in df.SCAN))
            else:
                sub_channels = channels
            dfs = {}
            for channel in sub_channels:
                dfs[channel] = df.assign(IMAGE=ims.apply(itemgetter(channel)))
            df = pd.concat(dfs, names=["CHANNEL", "CI"]).reset_index()
            spots.value = (df
                           .pivot_table('IMAGE', rows, cols, _join_images)
                           .applymap(_pil_to_html)
                           .to_html(escape=False, index_names=False))

        self.data.on('selection_changed', callback)
        return spots

    def structure(self, size=(175, 175), parse=Chem.MolFromSmiles, name='SMILES'):
        structure = HTML()

        def callback(_, df):
            df = df.get_selected_df()
            structure.value = ''
            mol = parse(str(df.iloc[0, df.columns.get_loc(name)]))
            if mol:
                drawer = rdMolDraw2D.MolDraw2DSVG(*size)
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                drawer.FinishDrawing()
                structure.value = drawer.GetDrawingText()

        self.data.on('selection_changed', callback)
        return structure

    @wraps(pd.DataFrame.to_csv)
    def to_csv(self, *args, **kwargs):
        return self.data.get_changed_df().to_csv(*args, **kwargs)


qgrid_options = {
    # SlickGrid options
    'forceFitColumns': False,  ##
    'defaultColumnWidth': 80,  ##
    'enableColumnReorder': True,  ##
    'autoEdit': False,  # ??

    # Qgrid options
    'maxVisibleRows': 6,  ##
    'minVisibleRows': 1,  ##
    'filterable': False,  ##
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

