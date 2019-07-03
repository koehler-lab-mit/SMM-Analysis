import io

import ipywidgets as widgets
import numpy as np
import qgrid
from IPython.display import display, SVG
from PIL import Image, ImageOps
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from skimage.exposure import rescale_intensity


# logger = open('/Users/rob/Desktop/logging2.txt', 'w')
# def write(str):
#     logger.write(str+'\n')
#     logger.flush()


class SpotDisplay(widgets.Output):

    def __init__(self, scans, channel, scale=2, size=(150, 150)):
        self.scans = scans
        self.channel = channel
        self.scale = scale
        self.size = size
        super().__init__()

    def display_spots(self, event=None, widg=None, df=None):
        if event:
            df = widg.get_changed_df().iloc[event['new']]
        hboxes = []
        for barcode, group in df.groupby('BARCODE'):
            hboxes.append(self._get_spots(group))
        self.clear_output()
        with self:
            display(widgets.VBox(hboxes))

    def _get_spots(self, df):
        hb = []
        for row in df.itertuples():
            hb.append(self._get_spot(row.Y, row.X, row.DIA, self.scans[row.BARCODE]))
        return widgets.HBox(hb)

    def _get_spot(self, y, x, dia, scan):
        r = dia * self.scale / 2
        im = scan[self.channel][int(y - r):int(y + r), int(x - r):int(x + r)]
        im = np.sqrt(im)
        im = rescale_intensity(np.sqrt(im), out_range=np.uint8).astype(np.uint8)
        im = Image.fromarray(im).convert(mode='L').resize(self.size)
        im = ImageOps.colorize(im, [0, 0, 0], colors[self.channel])
        b = io.BytesIO()
        im.save(b, 'PNG')
        return widgets.Image(value=b.getvalue(), type='png')


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
