import unittest

from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon

from geo.vectors import split_multiparts, merge_polygons, remove_interior_rings, get_extent


class TestGeometryFunctions(unittest.TestCase):

    def test_split_multiparts(self):
        # Test geometries
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
        multi_poly = MultiPolygon([poly1, poly2])

        line1 = LineString([(0, 0), (3, 3)])
        line2 = LineString([(1, 1), (4, 4)])
        multi_line = MultiLineString([line1, line2])

        # Test split_multiparts function with MultiPolygon
        single_parts_poly = split_multiparts(multi_poly)

        self.assertEqual(len(single_parts_poly), 2, "Expected 2 single-part geometries after splitting MultiPolygon")

        # Test split_multiparts function with MultiLineString
        single_parts_line = split_multiparts(multi_line)

        self.assertEqual(len(single_parts_line), 2, "Expected 2 single-part geometries after splitting MultiLineString")

    def test_merge_polygons(self):
        # Test geometries
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
        poly3 = Polygon([(5, 5), (8, 5), (8, 8), (5, 8)])
        poly4 = Polygon([(6, 6), (9, 6), (9, 9), (6, 9)])

        # Merge polygons without splitting multiparts
        merged_geometries_multiparts = merge_polygons([poly1, poly2, poly3, poly4], multiparts=True)

        self.assertEqual(len(merged_geometries_multiparts), 2, "Expected 2 multi-part geometries after merging")

    def test_remove_interior_rings(self):
        # test data
        polygon_with_hole = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
                                    [((2, 2), (2, 8), (8, 8), (8, 2), (2, 2))])

        polygon_without_hole = remove_interior_rings([polygon_with_hole])
        self.assertEqual(len(polygon_without_hole[0].interiors), 0, "Expected no interior rings after removing them")


    def test_get_extent(self):
        # Crear geometrías de prueba
        geom1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        geom2 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])

        # Calcular el extent utilizando la función get_extent
        extent = get_extent([geom1, geom2])

        # Verificar que el extent sea correcto
        expected_extent = (0.0, 0.0, 4.0, 4.0)
        self.assertEqual(extent.bounds, expected_extent, "Expected extent to match the coordinates of the combined geometries")


if __name__ == '__main__':
    unittest.main()
