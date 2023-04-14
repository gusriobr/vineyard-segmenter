import os
import sqlite3
import unittest

from shapely.geometry import Polygon

from geo.spatialite import create_connection, create_spatialite_table, list_all, remove_by_geometry, \
    insert_polygons


class TestSpatialite(unittest.TestCase):
    def setUp(self):
        self.db_path = 'test.sqlite'

    def test_create_connection(self):
        conn = create_connection(self.db_path)
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()

    def test_create_spatialite_table(self):
        table_name = 'test_table'
        table_features_sql = f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)"
        create_spatialite_table(self.db_path, table_name, table_features_sql)
        results = list_all(self.db_path, f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        self.assertIn((table_name,), results)

    def test_remove_by_extent(self):
        layer_name = 'test_layer'
        geometry_column = 'geometry'
        extent = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        conn = create_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE {layer_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, {geometry_column} POLYGON)")
        cursor.execute(
            f"INSERT INTO {layer_name} ({geometry_column}) VALUES (ST_GeomFromText('{extent.wkt}', 4258))")
        conn.commit()
        conn.close()
        remove_by_geometry(self.db_path, layer_name, geometry_column, extent)
        results = list_all(self.db_path, f"SELECT COUNT(*) FROM {layer_name}")
        self.assertEqual(results, [(0,)])

    def test_insert_polygons(self):
        layer_name = 'test_layer'
        polygons = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
            Polygon([(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)])
        ]
        create_spatialite_table(self.db_path, layer_name,
                                f"CREATE TABLE {layer_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, geometry POLYGON)")
        insert_polygons(self.db_path, layer_name, polygons)
        conn = create_connection(self.db_path)
        cursor = conn.cursor()
        results = cursor.execute(f"SELECT COUNT(*) FROM {layer_name}").fetchall()
        self.assertEqual(results, [(len(polygons),)])
        conn.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)


if __name__ == '__main__':
    unittest.main()
