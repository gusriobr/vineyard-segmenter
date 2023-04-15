import logging

from shapely.geometry import MultiLineString, Polygon, MultiPoint
from shapely.geometry import MultiPolygon


def split_multiparts(geometries):
    """
    Parts the multiparts of the given geometries list to create single geometries
    :param geometries:
    :return:
    """
    if not isinstance(geometries, list):
        geometries = [geometries]
    merged_geometries = []
    for geometry in geometries:
        if isinstance(geometry, MultiPolygon) or isinstance(geometry, MultiLineString):
            for single_part in geometry.geoms:
                merged_geometries.append(single_part)
        else:
            merged_geometries.append(geometry)
    return merged_geometries


def merge_polygons(shapely_geometries, multiparts=False):
    if not isinstance(shapely_geometries, list):
        shapely_geometries = [shapely_geometries]
    # Encontrar las geometrías solapadas y unirlas
    merged_geometries = []
    processed = set()
    for i, sg1 in enumerate(shapely_geometries):
        if i not in processed:
            merged = sg1
            for j, sg2 in enumerate(shapely_geometries[i + 1:], start=i + 1):
                if j not in processed and merged.intersects(sg2):
                    merged = merged.union(sg2)
                    processed.add(j)
            processed.add(i)
            merged_geometries.append(merged)

    if not multiparts:
        merged_geometries = split_multiparts(merged_geometries)
    return merged_geometries


def remove_interior_rings(geometries, fail_if_error=False):
    if not isinstance(geometries, list):
        geometries = [geometries]

    filtered = []
    for geometry in geometries:
        if isinstance(geometry, Polygon):
            # Crear un nuevo polígono con solo el anillo exterior
            filtered.append(Polygon(geometry.exterior))
        elif geometry.is_empty:
            # Si la geometría está vacía, devolverla tal como está
            filtered.append(geometry)
        else:
            msg = "La función solo admite geometrías de tipo Polygon."
            if fail_if_error:
                raise ValueError(msg)
            logging.error(msg)
    return filtered


def get_extent(shapely_geometries):
    """
    Recib
    :param geometries:
    :return: shapely Polygon object for the bounding box
    """
    # Crear un objeto MultiPoint a partir de las coordenadas de todas las geometrías
    all_coords = [coord for geom in shapely_geometries for coord in geom.exterior.coords]
    multi_point = MultiPoint(all_coords)

    # Obtener las coordenadas mínimas y máximas de todas las geometrías
    min_x, min_y, max_x, max_y = multi_point.bounds
    return Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y)])



