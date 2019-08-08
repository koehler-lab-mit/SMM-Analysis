import io
import logging
from itertools import accumulate, repeat, cycle, count

import ipywidgets as widgets
import numpy as np
import qgrid
from IPython.display import display, SVG
from PIL import Image, ImageOps, ImageFont, ImageDraw
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from skimage.exposure import rescale_intensity

logger = logging.getLogger(__name__)


def label_image(image, label, side='top', spacing=0,
                font=ImageFont.truetype("/Users/rob/PycharmProjects/SMM-Analysis/src/RobotoCondensed-Regular.ttf", 16)):
    width, height = image.size
    text_img = Image.new('RGB', font.getsize(label), (255, 255, 255, 0))
    ImageDraw.Draw(text_img).text((0, 0), label, font=font, fill=(0, 0, 0))
    if side == 'top' or side == 'bottom':
        text_width, text_height = text_img.size
        labeled_image = Image.new('RGB', (width, height + text_height + spacing), (255, 255, 255, 0))
        xtext = max(int((width - text_width) / 2), 0)
        xim = 0
        if side == 'top':
            ytext = 0
            yim = text_height + spacing
        else:
            yim = 0
            ytext = height + spacing
    elif side == 'left' or side == 'right':
        text_img = text_img.rotate(-90)
        text_width, text_height = text_img.size
        labeled_image = Image.new('RGB', (width + text_width + spacing, height), (255, 255, 255, 0))
        ytext = max(int((height - text_height) / 2), 0)
        yim = 0
        if side == 'left':
            xtext = 0
            xim = text_width + spacing
        else:
            xim = 0
            xtext = width + spacing
    else:
        raise ValueError("'side' must be one of (top, bottom, left, right), not %s" % side)
    labeled_image.paste(text_img, (xtext, ytext))
    labeled_image.paste(image, (xim, yim))
    return labeled_image


def combine_images(images, layout='horizontal', spacing=10):
    n_images = len(images)
    widths = [i.width for i in images]
    heights = [i.height for i in images]
    if layout == 'horizontal':
        width = sum(widths) + spacing * (n_images - 1)
        height = max(heights)
        x = [0] + [w + spacing * (i + 1) for i, w in enumerate(accumulate(widths))]
        y = repeat(0, n_images)
    elif layout == 'vertical':
        width = max(widths)
        height = sum(heights) + spacing * (n_images - 1)
        x = repeat(0, n_images)
        y = [0] + [h + spacing * (i + 1) for i, h in enumerate(accumulate(heights))]
    else:
        raise ValueError("Direction must be horizontal or vertical, not %s" % layout)
    img = Image.new('RGB', (width, height), (255, 255, 255, 0))
    for image, ix, iy in zip(images, x, y):
        img.paste(image, (ix, iy))
    return img


class SpotDisplay(widgets.Image):
    def __init__(self, channel, zoom=2, size=(150, 150), group_by='BARCODE', auto_contrast=True):
        self.channel = channel
        self.zoom = zoom
        self.size = size
        self.group_by = [group_by] if isinstance(group_by, str) else group_by
        self.auto_contrast = auto_contrast
        super().__init__(type='png')

    def get_spot_image(self, scan, x, y, dia):
        scan = scan[self.channel]
        dia *= self.zoom
        x = int((x - scan.x_offset - dia / 2) / scan.resolution)
        y = int((y - scan.y_offset - dia / 2) / scan.resolution)
        dia = int(dia / scan.resolution)
        if self.auto_contrast:
            image = rescale_intensity(np.sqrt(scan.data[y:y + dia, x:x + dia]), out_range=np.uint8).astype(np.uint8)
        else:
            image = np.sqrt(scan.data[y:y + dia, x:x + dia]).astype(np.uint8)
        image = Image.fromarray(image).convert(mode='L').resize(self.size)
        if scan.channel:
            image = ImageOps.colorize(image, [0, 0, 0], colors[scan.channel])
        return image

    def show_spots_from_frame(self, df):
        df = df.copy()
        group_by = self.group_by.copy()
        df["IMAGE"] = df.apply(lambda x: self.get_spot_image(x.SCAN, x.X, x.Y, x.DIA), axis=1)
        logger.error(group_by)
        images = df.set_index(group_by)["IMAGE"]
        image_layouts = cycle(['horizontal', 'vertical'])
        label_directions = cycle(['left', 'up', 'right', 'down'])
        spacing_size = (int(i / 2) * 5 for i in count(2))
        for layout, label_direction, spacing in zip(image_layouts, label_directions, spacing_size):
            if group_by:
                images = images.groupby(group_by).apply(lambda x: combine_images(x, layout, spacing))
                group_by.pop()
            else:
                return combine_images(images, layout, spacing)

    def connect_qgrid(self, qgrid_df, lookup_func=None):
        def callback(event, widget):
            df = widget.get_changed_df().iloc[event['new']]
            if lookup_func:
                df = lookup_func(df)
            image = self.show_spots_from_frame(df)
            b = io.BytesIO()
            image.save(b, 'PNG')
            self.value = b.getvalue()
        qgrid_df.on('selection_changed', callback)


class TableDisplay(widgets.Output):
    def __init__(self, data):
        self.data = data
        self.height = 150
        self.width = 150
        super().__init__()
        self.struct = widgets.Output(layout=widgets.Layout(width='20%'))
        self.qgrid_widget = qgrid.show_grid(
            self.data,
            show_toolbar=False,
            grid_options=qgrid_default_grid_options,
            column_options=qgrid_default_column_options)
        self.qgrid_widget.on('selection_changed', self.update_struct)
        grid_out = widgets.Output(layout=widgets.Layout(width='80%'))
        with grid_out:
            display(self.qgrid_widget)
        with self:
            display(widgets.HBox((grid_out, self.struct)))

    def on(self, event, handler):
        self.qgrid_widget.on(event, handler)

    def update_struct(self, event, widg):
        smiles = str(widg.get_changed_df().iloc[event['new'][0]].SMILES)
        mol = Chem.MolFromSmiles(smiles)
        drawer = rdMolDraw2D.MolDraw2DSVG(175, 175)
        if mol:
            mol = rdMolDraw2D.PrepareMolForDrawing(mol)
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        self.struct.clear_output()
        with self.struct:
            display(SVG(drawer.GetDrawingText().replace("svg:", "")))

colors = {'635': [255, 0, 0], '594': [255, 128, 0], '532': [0, 255, 0], '488': [0, 75, 255]}

qgrid_default_grid_options = {
    # SlickGrid options
    'fullWidthRows': True,
    'syncColumnCellResize': True,
    'forceFitColumns': False,
    'defaultColumnWidth': 80,
    'rowHeight': 28,
    'enableColumnReorder': True,
    'enableTextSelectionOnCells': True,
    'editable': True,
    'autoEdit': False,
    'explicitInitialization': True,

    # Qgrid options
    'maxVisibleRows': 6,
    'minVisibleRows': 1,
    'sortable': True,
    'filterable': True,
    'highlightSelectedCell': True,
    'highlightSelectedRow': True
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
