import unittest

import SMM.IO


class MyTestCase(unittest.TestCase):

    def test_load_atf(self):
        path = '../../../data/BlockAlign.gpr'
        atf = SMM.IO.load_atf(path)
        self.assertEqual(len(atf.headers), 32)
        self.assertEqual(atf.data.shape, (12288, 56))

    def test_load_gal(self):
        gal = SMM.IO.load_gal('../../../data/sample.gal', map_blocks=True)
        self.assertEqual(gal.data.shape, (12288, 8))
        self.assertEqual(len(gal.headers), 55)

    def test_load_scan(self):
        path = '../../../data/sample.tif'
        tifs = SMM.IO.load_tif(path, cache='all')
        channels = [x.channel for x in tifs]
        self.assertIn('532', channels)
        self.assertIn('635', channels)
        self.assertEqual(tifs[0].data.shape, (7180, 2180))


if __name__ == '__main__':
    unittest.main()
