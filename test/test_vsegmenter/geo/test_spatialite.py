import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

from shapely.geometry import Polygon

from geo.spatialite import create_connection, create_spatialite_table, list_all, remove_by_geometry, \
    insert_polygons, intersect_layers, get_srid


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


class TestIntersectDBs(unittest.TestCase):

    def setUp(self):
        self.db_file1 = NamedTemporaryFile(suffix=".sqlite", delete=False)
        self.db_file2 = NamedTemporaryFile(suffix=".sqlite", delete=False)
        self.polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        ]
        self.srid = 4258

        conn1 = create_connection(self.db_file1.name)
        conn2 = create_connection(self.db_file2.name)

        # Crear tabla1 con datos
        conn1.execute(f"""
            CREATE TABLE tabla1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                geometry GEOMETRY
            );
        """)
        for i, polygon in enumerate(self.polygons):
            conn1.execute(f"""
                INSERT INTO tabla1 (name, geometry)
                VALUES ('polygon{i}', ST_GeomFromText('{polygon.wkt}', {self.srid}));
            """)

        # Crear tabla2 con datos
        conn2.execute(f"""
            CREATE TABLE tabla2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                geometry GEOMETRY
            );
        """)
        conn2.execute(f"""
            INSERT INTO tabla2 (name, geometry)
            VALUES ('polygon3', ST_GeomFromText('{Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]).wkt}', {self.srid}));
        """)
        conn1.commit()
        conn2.commit()
        conn1.close()
        conn2.close()

    def tearDown(self):
        Path(self.db_file1.name).unlink()
        Path(self.db_file2.name).unlink()

    def test_insert_dbs(self):
        result = intersect_layers(self.db_file1.name, "tabla1", "geometry",
                                  self.db_file2.name, "tabla2", "geometry")
        self.assertEqual(1, len(result), )
        self.assertEqual(1, result[0][0])  # ID de primer polígono en tabla1
        self.assertTrue(result[0][1].equals(self.polygons[0]))


class TestGetSrid(unittest.TestCase):
    def setUp(self):
        # Crea un archivo temporal para la BD
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db = self.temp_dir.name + '/test_db.sqlite'

        # Crea la BD SpatiaLite y una tabla con una columna de geometría
        conn = create_connection(self.test_db)
        conn.execute('SELECT InitSpatialMetadata(1)')
        conn.execute('CREATE TABLE test_layer (id INTEGER PRIMARY KEY, geometry POLYGON)')
        conn.execute('SELECT AddGeometryColumn("test_layer", "geometry", 4326, "POLYGON", "XY")')
        conn.execute(
            'INSERT INTO test_layer (geometry) VALUES (ST_GeomFromText("POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))", 4326))')
        conn.commit()
        conn.close()

    def tearDown(self):
        # Borra el archivo temporal después del test
        self.temp_dir.cleanup()

    def test_get_srid(self):
        # Define los valores de entrada
        db_file = self.test_db
        layer = 'test_layer'
        geometry = 'geometry'

        # Ejecuta la función y comprueba el resultado
        result = get_srid(db_file, layer, geometry)
        self.assertEqual(result, [4326])


if __name__ == '__main__':
    unittest.main()
