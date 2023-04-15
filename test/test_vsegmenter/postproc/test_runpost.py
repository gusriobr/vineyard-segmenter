import unittest
from shapely.geometry import Polygon

from postproc.run_post import simplify_polygon


class TestSimplifyPolygon(unittest.TestCase):
    def test_simplify_polygon(self):
        # Test case 1: Polygon with multiple vertices
        polygon1 = Polygon([(0,0), (0,1), (1,1), (1,0)])
        simplified_polygon1 = simplify_polygon(polygon1)
        self.assertEqual(simplified_polygon1.wkt, 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))')

        # Test case 2: Polygon with fewer vertices
        polygon2 = Polygon([(0,0), (0,2), (2,2), (2,0)])
        simplified_polygon2 = simplify_polygon(polygon2)
        self.assertEqual(simplified_polygon2.wkt, 'POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))')

        # Test case 3: Polygon with no vertices
        polygon3 = Polygon([])
        simplified_polygon3 = simplify_polygon(polygon3)
        self.assertEqual(simplified_polygon3.wkt, 'GEOMETRYCOLLECTION EMPTY')

if __name__ == '__main__':
    unittest.main()
