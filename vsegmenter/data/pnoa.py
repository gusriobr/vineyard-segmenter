import os
import rasterio
import geopandas as gpd
import sqlite3
from shapely.geometry import box

import cfg
import logging

cfg.configLog()


def create_spatialite_table(sqlite_path):
    conn = create_connection(sqlite_path)

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pnoa_tiles (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            year NUMERIC NOT NULL,
            h50 TEXT NOT NULL,
            cell TEXT NOT NULL,
            huso TEXT NOT NULL,
            filepath TEXT NOT NULL,
            provmunis TEXT NOT NULL,
            ines TEXT NOT NULL,
            geom GEOMETRY NOT NULL
        )
    """)
    cursor.execute("SELECT InitSpatialMetaData(1);")
    cursor.execute("SELECT AddGeometryColumn('pnoa_tiles', 'geom', 4326, 'POLYGON', 'XY');")
    conn.commit()


def create_connection(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    return conn


def insert_into_pnoa_tiles(conn, data, geom):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pnoa_tiles WHERE FILENAME = ?", (data["filename"],))
    cursor.execute("""
        INSERT INTO pnoa_tiles (filename,year,h50,cell,huso,filepath,provmunis,ines,geom)
        VALUES (?,?,?,?,?,?,?,?,GeomFromText(?, 4326));
    """, (data["filename"], data["year"], data["h50"], data["cell"], data["huso"], data["filepath"], data["provmunis"], data["ines"],
          geom.wkt))
    conn.commit()


def locate_raster_files(raster_dir):
    paths = []
    for root, _, files in os.walk(raster_dir):
        for filename in files:
            if filename.endswith(".tiff") or filename.endswith(".tif"):
                filepath = os.path.join(root, filename)
                paths.append(filepath)
    return sorted(paths)


def get_raster_polygon(filepath, srid=4258):
    """
    Returns raster extends represented as a polygon in 4258 SRID
    :param filepath:
    :return:
    """
    with rasterio.open(filepath) as src:
        origin_srid = src.crs
        bounds = src.bounds
        raster_polygon = box(*bounds)

    if origin_srid != srid:
        # Convertir el polígono del raster a SRID 4258
        raster_gdf = gpd.GeoDataFrame({'geometry': [raster_polygon]}, crs=origin_srid)
        raster_gdf = raster_gdf.to_crs(epsg=srid)
        raster_polygon = raster_gdf.iloc[0].geometry
    return raster_polygon


def get_intersecting_munis(raster_polygon, municipalities_gdf):
    intersecting_municipalities = municipalities_gdf[municipalities_gdf.intersects(raster_polygon)]
    provmuni_list = sorted(intersecting_municipalities.C_PROVMUN.tolist())
    ine_list = sorted(intersecting_municipalities.C_PROVMUNINE.tolist())
    return provmuni_list, ine_list


def get_raster_attributes(filepath, provmuni_list, ine_list):
    data = {}
    filename = os.path.basename(filepath)
    # 'PNOA', 'CYL', '2020', '25cm', 'OF', 'etrsc', 'rgb', 'hu30', 'h05', '0345', '3-2.tif'
    parts = filename.split("_")
    data["filename"] = filename
    data["cell"] = parts[-1].split(".")[0]
    data["h50"] = parts[-2]
    data["huso"] = parts[-4]
    data["year"] = parts[2]
    data["filepath"] = filepath
    data["provmunis"] = ",".join(provmuni_list)
    data["ines"] = ",".join(ine_list)
    return data


def process_rasters(raster_dir, output_path, muni_file):
    # Cargar los municipios
    municipalities_gdf = gpd.read_file(muni_file)

    create_spatialite_table(output_path)

    raster_files = locate_raster_files(raster_dir)

    # Iterar sobre los archivos TIFF en el directorio
    conn = create_connection(output_path)
    for idx, filepath in enumerate(raster_files):
        logging.info(f"Processing raster {idx+1} of {len(raster_files)} file: {filepath} ")
        # get raster bbox
        raster_polygon = get_raster_polygon(filepath)

        # Obtener la lista de municipios que intersectan con el polígono
        provmuni_list, ine_list = get_intersecting_munis(raster_polygon, municipalities_gdf)

        # # Insertar en la tabla pnoa_tiles
        rel_path = os.path.relpath(filepath, raster_dir)
        attributes = get_raster_attributes(rel_path, provmuni_list, ine_list)
        insert_into_pnoa_tiles(conn, attributes, raster_polygon)
        logging.info(f"Inserted record with municipalities: {provmuni_list}")

    conn.close()
    logging.info(f"Process successfully finished, inserted {len(raster_files)} records")


if __name__ == '__main__':
    # Ejemplo de uso
    raster_directory = "/media/gus/data/rasters/aerial/pnoa/2020"
    muni_file = '/media/gus/data/cartography/municipios/provmun_4258_2018.sqlite'
    index_file = "/media/gus/data/rasters/aerial/pnoa/2020/index.sqlite"
    process_rasters(raster_directory, index_file, muni_file)
