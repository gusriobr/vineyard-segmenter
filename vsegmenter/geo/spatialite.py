import sqlite3

import shapely.wkb
from shapely.wkb import dumps


def create_connection(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    return conn


def create_spatialite_table(sqlite_path, table_name, table_features_sql, geomtry_col='geom', srid=4326):
    conn = create_connection(sqlite_path)
    cursor = conn.cursor()
    cursor.execute(table_features_sql)
    cursor.execute("SELECT InitSpatialMetaData(1);")
    cursor.execute(f"SELECT AddGeometryColumn('{table_name}', '{geomtry_col}', {srid}, 'POLYGON', 'XY');")
    conn.commit()


def list_all(db_file, sql, apply_function=None):
    conn = create_connection(db_file)
    results = conn.execute(sql).fetchall()
    conn.close()
    if apply_function:
        results = [apply_function(row) for row in results]
    return results


def list_by_geometry(db_file, layer_name, geometry_column, extent, srid=4258):
    sql = f"SELECT ST_AsBinary({geometry_column}) FROM {layer_name} WHERE ST_Intersects({geometry_column}, ST_GeomFromText('{extent.wkt}', {srid}))"
    return list_all(db_file, sql, lambda x: shapely.wkb.loads(x[0]))


def remove_by_geometry(db_file, layer_name, geometry_column, extent, srid=4258):
    """
    Removes polygons that intersect with the given extent in the proposed layer
    """
    conn = create_connection(db_file)
    sql = f"DELETE FROM {layer_name} WHERE ST_Intersects({geometry_column}, ST_GeomFromText('{extent.wkt}', {srid}))"
    conn.execute(sql)
    conn.commit()
    conn.close()


def insert_polygons(db_file, layer_name, geometry_column, polygons, srid=4258):
    """
    :param db_file:
    :param layer_name:
    :param geometry_column:
    :param polygons: shapely polygons
    :param srid:
    :return:
    """
    if not isinstance(polygons, list):
        polygons = [polygons]
    # Connect to the Spatialite database
    conn = create_connection(db_file)

    # Start a transaction
    cursor = conn.cursor()
    cursor.execute("BEGIN") # init transaction

    # Prepare the INSERT statement for the layer
    sql = f"INSERT INTO {layer_name} ({geometry_column}) VALUES (ST_GeomFromText(?, {srid}))"

    # Loop through the polygons and insert them into the layer
    for polygon in polygons:
        cursor.execute(sql, (polygon.wkt,))
    # Commit the transaction and close the database connection
    conn.commit()
    conn.close()
