import cfg
import geo.spatialite as spt

cfg.configLog()
import logging

def update_extraction_info(sample_file, pnoa_index):
    """
    Updates extraction polygons assigning to each extraction polygon its intersecting pnoa tile and sets the id
    :param sample_file: 
    :param pnoa_index: 
    :return: 
    """
    conn = spt.create_connection(sample_file)
    # attach pnoa index
    conn.execute("ATTACH DATABASE ? AS db2", (pnoa_index,))

    # check srids
    t1_srid = spt.get_srid(sample_file, 'extractions', 'geometry')
    t2_srid = spt.get_srid(pnoa_index, 'pnoa_tiles', 'geom')
    st_intersect = f"ST_Intersects(t1.geom, t2.geometry)"
    if t1_srid != t2_srid:
        st_intersect = f"ST_Intersects(t1.geometry, st_transform(t2.geom, {t1_srid}))"

    query = f"""
               SELECT t1.id, t2.filename, t2.h50, t2.cell
               FROM extractions t1, db2.pnoa_tiles t2 
               WHERE {st_intersect}
           """
    logging.info(query)
    results = conn.execute(query).fetchall()

    # udpate the extraction info
    lst_data = []
    extraction_ids = set()
    duplicated_ids = set()
    for row in results:
        # extraction filename, pnoa_tile, id
        extraction_id = row[0]
        update_data = [f"{row[2]}_{row[3]}__{extraction_id}.tif", row[1], extraction_id]
        if extraction_id not in extraction_ids:
            extraction_ids.add(extraction_id)
        else:
            duplicated_ids.add(extraction_id)
        extraction_ids.add(row[0])
        lst_data.append(update_data)
    # check the extraction features and contained in just one tile
    if duplicated_ids:
        msg = f"This extractions intersect with more than one PNOA tile: {duplicated_ids}"
        raise ValueError(msg)
    logging.info("Updating extraction features info to set pnoa_tile filename and extraction filename.")
    for data in lst_data:
        query = "update extractions set filename = ?, pnoa_tile=? where id = ?"
        conn.execute(query, data)
    conn.commit()

    conn.close()

if __name__ == '__main__':

    sample_file = cfg.resources("dataset/samples.sqlite")
    pnoa_index = cfg.pnoa("pnoa_index.sqlite")
    pnoa_folder = cfg.PNOA_BASE_FOLDER

    update_extraction_info(sample_file, pnoa_index)
