import unittest

import cfg
from vsegmenter.image.raster import get_raster_bbox


class TestRaster(unittest.TestCase):
    def setUp(self):
        self.raster_file = cfg.test_resource("test_raster.tiff")

    def test_get_raster_bbox(self):
        # Compute the bounding box of the input raster
        bbox, csr = get_raster_bbox(self.raster_file)
        # Verify that the bounding box coordinates are reasonable
        self.assertGreater(bbox[2], bbox[0])
        self.assertGreater(bbox[3], bbox[1])
        self.assertEqual(len(bbox), 4)


if __name__ == '__main__':
    unittest.main()
