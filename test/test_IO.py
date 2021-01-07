import unittest

import smm_io.Scan


class MyTestCase(unittest.TestCase):

    def test_load_atf(self):
        atf = smm_io.Scan.load_atf('../data/BlockAlign.gpr')
        self.assertEqual(len(atf.attrs['headers']), 32)
        self.assertEqual(atf.shape, (12288, 56))

    def test_load_gal(self):
        gal = smm_io.Scan.load_gal('../data/sample.gal', map_blocks=True)
        self.assertEqual(gal.shape, (12288, 8))
        self.assertEqual(len(gal.attrs['headers']), 55)

    def test_load_scan(self):
        path = '../data/sample.tif'
        tifs = smm_io.Scan.load_tif('../data/sample.tif', cache='all')
        channels = [x.attrs.channel for x in tifs]
        self.assertIn('532', channels)
        self.assertIn('635', channels)
        self.assertEqual(tifs[0].as_numpy.shape, (7180, 2180))


if __name__ == '__main__':
    unittest.main()
