from base64 import b64encode
from functools import wraps
from io import BytesIO
from operator import itemgetter

import ipywidgets as widgets
import numpy as np
import pandas as pd
import qgrid
from IPython.display import display
from PIL import Image, ImageOps
from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import IPythonConsole

IPythonConsole.ipython_useSVG = True
rdDepictor.SetPreferCoordGen(True)
IPythonConsole.molSize = (175, 175)
rdBase.DisableLog('rdApp.*')

pd.set_option('display.max_colwidth', -1)

colors = {'635': [255, 0, 0],
          '594': [255, 128, 0],
          '532': [0, 255, 0],
          '488': [0, 75, 255]}


def set_structure_size(width, height):
    IPythonConsole.molSize = (width, height)


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


class TableDisplay(widgets.Output):

    def __init__(self, data):
        super().__init__()
        self.table = qgrid.show_grid(
            data,
            show_toolbar=False,
            grid_options=qgrid_default_grid_options,
            column_options=qgrid_default_column_options)
        if 'SMILES' in data:
            self.struct = widgets.Output(layout=widgets.Layout(width='20%'))
            self.table.on('selection_changed', self.update_struct)
            grid_out = widgets.Output(layout=widgets.Layout(width='80%'))
            with grid_out:
                display(self.table)
            with self:
                display(widgets.HBox((grid_out, self.struct)))
        else:
            with self:
                display(self.table)

    def show_spots(self, channels=None, rows=None, cols=None,
                   zoom=2, size=150, contrast=(20, 98), lookup_func=None):
        table = widgets.HTML()
        channels = [channels] if isinstance(channels, str) else channels
        rows = [rows] if isinstance(rows, str) else rows
        cols = [cols] if isinstance(cols, str) else cols
        if rows is None and cols is None:
            cols = ['ID']

        def callback(_, widget):
            df = widget.get_selected_df()
            if lookup_func:
                df = lookup_func(df)
            r = df.DIA * zoom / 2
            df = df.assign(y0=df.Y - r, y1=df.Y + r, x0=df.X - r, x1=df.X + r)
            ims = df.apply(_ImageFactory, axis=1, args=((size, size), contrast))
            if channels is None:
                sub_channels = set().union(*(x.keys() for x in df.SCAN))
            else:
                sub_channels = channels
            dfs = {}
            for channel in sub_channels:
                dfs[channel] = df.assign(IMAGE=ims.apply(itemgetter(channel)))
            df = pd.concat(dfs, names=["CHANNEL", "CI"]).reset_index()
            table.value = (df
                           .pivot_table('IMAGE', rows, cols, _join_images)
                           .applymap(_pil_to_html)
                           .to_html(escape=False, index_names=False))

        self.table.on('selection_changed', callback)
        return table

    @wraps(pd.DataFrame.to_csv)
    def to_csv(self, *args, **kwargs):
        return self.table.get_changed_df().to_csv(*args, **kwargs)

    def update_struct(self, _, df):
        mol = Chem.MolFromSmiles(str(df.get_selected_df().SMILES.iloc[0]))
        self.struct.clear_output()
        if mol:
            with self.struct:
                display(mol)


qgrid_default_grid_options = {
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

qgrid_default_column_options = {
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

