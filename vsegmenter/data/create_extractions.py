import os

import cfg
from geo.spatialite import create_connection, create_spatialite_table
from image.raster import get_raster_bbox

dataset_file = '/media/gus/workspace/wml/vineyard-segmenter/resources/dataset/samples.sqlite'
raster_folder = '/media/gus/data/viticola/datasets/segmenter/v5/extractions'

EXTRACTIONS_SRID = 4258
create_table_query = "CREATE TABLE extractions (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT)"
create_spatialite_table(dataset_file, table_name="extractions", table_features_sql=create_table_query,
                        geometry_col="geometry", drop_if_exists=True, srid=EXTRACTIONS_SRID)

with create_connection(dataset_file) as conn:
    for filename in os.listdir(raster_folder):
        polygon, csr = get_raster_bbox(os.path.join(raster_folder, filename))
        polygon.wkt
        # wkt_polygon = f"POLYGON(({left} {bottom}, {left} {top}, {right} {top}, {right} {bottom}, {left} {bottom}))"
        insert_query = f"""
                INSERT INTO extractions (filename, geometry) 
                VALUES ('{filename}', 
                    st_transform(ST_GeomFromText('{polygon.wkt}', {csr}), {EXTRACTIONS_SRID}))
                """
        conn.execute(insert_query)
