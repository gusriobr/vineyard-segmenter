import sqlite3

import shapely.wkb
from shapely.wkb import dumps
import logging


def create_connection(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    return conn


def create_spatialite_table(db_file, table_name, table_features_sql, geomtry_col='geom', srid=4326, drop_if_exists=False):
    conn = create_connection(db_file)
    if drop_if_exists:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"SELECT DiscardGeometryColumn('{table_name}', '{geomtry_col}')")
    cursor = conn.cursor()
    cursor.execute(table_features_sql)
    cursor.execute(f"SELECT AddGeometryColumn('{table_name}', '{geomtry_col}', {srid}, 'POLYGON', 'XY');")
    conn.commit()


def list_all(db_file, sql, apply_function=None, add_cols_desc = False):
    conn = create_connection(db_file)
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    try:
        if apply_function:
            results = [apply_function(row) for row in results]
        if not add_cols_desc:
            return results
        else:
            col_names = [description[0] for description in cursor.description]
            return results, col_names
    finally:
        cursor.close()
        conn.close()



def list_features(db_file, layer_name, geometry_column, where_filter = ""):
    if where_filter:
        where_filter = f"where {where_filter}"
    sql = f"SELECT t.*, ST_AsBinary({geometry_column}) as geometry FROM {layer_name}  t {where_filter}"
    result, col_names = list_all(db_file, sql, add_cols_desc=True)
    features = []
    for row in result:
        r_dict = {}
        for idx, col_name in enumerate(col_names):
            if col_name != geometry_column:
                r_dict[col_name] = row[idx]
        # add geometry
        r_dict[geometry_column] = shapely.wkb.loads(row[-1])
        features.append(r_dict)
    return features


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
    cursor.execute("BEGIN")  # init transaction

    # Prepare the INSERT statement for the layer
    sql = f"INSERT INTO {layer_name} ({geometry_column}) VALUES (ST_GeomFromText(?, {srid}))"

    # Loop through the polygons and insert them into the layer
    for polygon in polygons:
        cursor.execute(sql, (polygon.wkt,))
    # Commit the transaction and close the database connection
    conn.commit()
    conn.close()


def get_srid(db_file, layer, geometry):
    query = f"select srid from geometry_columns where f_table_name='{layer}' and f_geometry_column='{geometry}'"
    results = list_all(db_file, query)
    return results[0][0]


def intersect_layers(db_file1, layer1, geometry_column1, db_file2, layer2, geometry_column2):
    """
    Returns a list of tuples where each tuple represents a row from the intersection of two geometry layers stored
    in two SQLite database files.

    :param db_file1 (str): Path to the first SQLite database file.
    :param layer1 (str): Name of the table in the first database containing the first geometry layer.
    :param geometry_column1 (str): Name of the geometry column in the first table.
    :param db_file2 (str): Path to the second SQLite database file.
    :param layer2 (str): Name of the table in the second database containing the second geometry layer.
    :param geometry_column2 (str): Name of the geometry column in the second table.

    :return:     A list of tuples where each tuple represents a row from the intersection of the two geometry layers.
    The last element of each tuple is a shapely.geometry object representing the intersection of the two geometries.

    """
    conn = create_connection(db_file1)
    conn.execute("ATTACH DATABASE ? AS db2", (db_file2,))

    # check srids
    t1_srid = get_srid(db_file1, layer1, geometry_column1)
    t2_srid = get_srid(db_file2, layer2, geometry_column2)
    st_intersect = f"ST_Intersects(t1.{geometry_column1}, t2.{geometry_column2})"
    if t1_srid != t2_srid:
        st_intersect = f"ST_Intersects(t1.{geometry_column1}, st_transform(t2.{geometry_column2}, {t1_srid}))"

    query = f"""
            SELECT t1.*, ST_AsBinary(t1.{geometry_column1}) as geometry
            FROM {layer1} t1, db2.{layer2} t2 
            WHERE {st_intersect}
        """
    logging.info(query)
    with conn:
        results = conn.execute(query).fetchall()
        lst = []
        for row in results:
            r = list(row[:-2])
            r.append(shapely.wkb.loads(row[-1]))
            lst.append(r)
        return lst
