import unittest

from smm_io import Scan
import SMM_Widgets as sw
from pathlib import Path
import pandas as pd


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        p = Path('../data/TDP43')
        cls.data = pd.concat(
            {path.stem: Scan.load_gpr(path) for path in p.glob('*.gpr')},
            names=['BARCODE', 'BARCODE_INDEX']
        ).reset_index()
        cls.scans = Scan.load_tifs(p.glob('*.tif'))

    def test_structure(self):
        qg = sw.TableDisplay.show_grid(self.data.head(10))
        struct = qg.structure()
        self.assertEqual(len(struct.value), 0)
        qg.change_selection([2,3])
        self.assertGreater(len(struct.value), 0)



if __name__ == '__main__':
    unittest.main()
