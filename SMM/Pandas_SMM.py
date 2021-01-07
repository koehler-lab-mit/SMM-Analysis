import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle


@pd.api.extensions.register_dataframe_accessor('smm')
class MicroarrayAccessor:
    """
    Convenience functions for working with small molecule microarray data,
    stored in pandas DataFrames.
    Once this Accessor has been registered, functions can be called
    through object.name.function syntax.
    This is essentially a way to supplement DataFrames with new methods
    without requiring new subclasses.
    """

    needed_columns = {'BLOCK', 'COLUMN', 'ROW', 'NAME', 'ID', 'X', 'Y', 'DIA'}

    def __init__(self, dataframe):
        self._validate(dataframe)
        self._df = dataframe

    @staticmethod
    def _validate(obj):
        missing_columns = set.difference(MicroarrayAccessor.needed_columns, obj.columns)
        if missing_columns:
            raise AttributeError(f"DataFrame is missing {missing_columns}")
        if obj[['BLOCK', 'COLUMN', 'ROW']].duplicated().sum():
            raise AttributeError('DataFrame contains duplicated spots')

    def show_array(self):
        fig, ax = plt.subplots()
        arr = self._df[['X', 'Y', 'DIA']].to_numpy()
        arr[:, 2] = arr[:, 2] / 2
        circles = [Circle((x, y), r) for x, y, r in arr]
        circles = PatchCollection(circles, alpha=0.5)
        circles.set_color('black')
        ax.add_collection(circles)
        ax.invert_yaxis()
        ax.autoscale_view()
        plt.show()

    def fetch_pixels(self, scan, gap_pixels=2, background_size=3):
        pass

    def blocks(self):
        """
        Fits a grid to the observed data, assuming that blocks are numbered left
        to right, and then top to bottom. This is necessary because deleted blocks
        cause re-labelling of the subsequent blocks in Genepix, so they can't
        be used to determine the grid.
        :return: A DataFrame mapping blocks to columns and rows
        """
        blocks = self._df.groupby('BLOCK')[['X', 'Y']].mean().sort_index()
        x = blocks.X.diff()
        blocks['ROW'] = (x < 0).cumsum() + 1
        blocks['COLUMN'] = blocks.groupby('ROW')['X'].rank().astype(int)
        return blocks

    def check_sanity(self):
        # TODO Implement function to check if the grid alignment is reasonable
        pass


def fit_array(df):
    df = pd.DataFrame(dict(
        BLOCK=df.BLOCK.astype('category'),
        COLUMN=df.COLUMN - 1,
        ROW=df.ROW - 1,
        X=df.X, Y=df.Y
    ))
    p = patsy.dmatrix('BLOCK + BLOCK:COLUMN + BLOCK:ROW - 1', df)
    fit = sm.OLS(df[['X', 'Y']], p).fit()
    resid = np.sqrt((fit.resid ** 2).sum(1))
    return fit, resid
