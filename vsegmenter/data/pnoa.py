"""
Script para creación del índice de tiles pnoa
"""

import os
import rasterio
import geopandas as gpd
from shapely.geometry import box

import cfg
import logging

from geo.spatialite import create_connection, create_spatialite_table

cfg.configLog()

TABLE_PNOA_TILES = {"name": "pnoa_tiles",
                    "sql": """
        CREATE TABLE IF NOT EXISTS pnoa_tiles (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            year NUMERIC NOT NULL,
            h50 TEXT NOT NULL,
            cell TEXT NOT NULL,
            huso TEXT NOT NULL,
            filepath TEXT NOT NULL,
            provmunis TEXT NOT NULL,
            ines TEXT NOT NULL
        )
    """}


def insert_into_pnoa_tiles(conn, data, geom):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pnoa_tiles WHERE FILENAME = ?", (data["filename"],))
    cursor.execute("""
        INSERT INTO pnoa_tiles (filename,year,h50,cell,huso,filepath,provmunis,ines,geom)
        VALUES (?,?,?,?,?,?,?,?,GeomFromText(?, 4326));
    """, (data["filename"], data["year"], data["h50"], data["cell"], data["huso"], data["filepath"], data["provmunis"],
          data["ines"],
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


def create_pnoa_index(raster_dir, output_path, muni_file):
    # Cargar los municipios
    municipalities_gdf = gpd.read_file(muni_file)

    create_spatialite_table(output_path, TABLE_PNOA_TILES["name"], TABLE_PNOA_TILES["sql"], geomtry_col="geom",
                            srid=4326)

    raster_files = locate_raster_files(raster_dir)

    # Iterar sobre los archivos TIFF en el directorio
    conn = create_connection(output_path)
    for idx, filepath in enumerate(raster_files):
        logging.info(f"Processing raster {idx + 1} of {len(raster_files)} file: {filepath} ")
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


def create_pnoa_list_for_geo(index_file, geometry):
    """
    Creates the pnoa file list for a given geometry
    """
    conn = create_connection(index_file)

    cursor = conn.cursor()
    cursor.execute("""
        SELECT filepath FROM pnoa_tiles where st_intersects(geom, GeomFromText(?, 4258))
    """, (geometry.wkt,))
    return [r[0] for r in cursor.fetchall()]


def create_pnoa_list_for_municipalities(index_file, muni_list):
    """
    Creates the pnoa file list for a given list of municipalities
    """
    conn = create_connection(index_file)
    tile_list = []
    for muni_id in muni_list:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT filepath FROM pnoa_tiles where provmunis like '%{muni_id}%'
        """)
        tile_list.extend([r[0] for r in cursor.fetchall()])
    return tile_list


def get_do_geomety_by_name(do_name, srid=None):
    """
    :param do_name: do name
    :return: shapely geometry
    """
    do_file = cfg.cartography("tematicos/figuras_calidad_vino/20200813_DO_VINO.shp")
    df = gpd.read_file(do_file)
    df["D_DO"] = df.D_DO.fillna("---")
    geo_col = df[df.D_DO.str.contains(do_name.upper())].geometry
    if srid:
        geo_col = geo_col.to_crs(4258)
    return geo_col.iloc[0]


if __name__ == '__main__':
    # Ejemplo de uso
    # raster_directory = "/media/gus/data/rasters/aerial/pnoa/2020"
    raster_directory = "/media/cartografia/01_Ortofotografia/2020/RGB"
    # muni_file = '/media/gus/data/cartography/municipios/provmun_4258_2018.sqlite'
    muni_file = cfg.cartography("provmun_4258_2018.sqlite")
    index_file = cfg.cartography("/workspaces/cartography/pnoa_index.sqlite")

    ########################
    ### creación indice de tiles pnoa
    ########################
    # process_rasters(raster_directory, index_file, muni_file)

    ########################
    # creación de ficheros de índices pnoa a partir de geometrias/listado municipios de DO's
    ########################
    for do_name in ["ribera", "arlanza", "toro"]:
        # utilizando geometría do
        do_geo = get_do_geomety_by_name("ribera", srid="4258")
        raster_files = create_pnoa_list_for_geo(index_file, do_geo)
        raster_files = [cfg.pnoa(filename) for filename in raster_files]

        # utilizando municipios
        # raster_files = create_pnoa_list_for_municipalities(index_file, ["05132","05014","05132"])

        with open(cfg.resources(f"pnoa_{do_name}.txt"), 'w') as f:
            for filename in raster_files:
                f.write(f"{filename}\n")
