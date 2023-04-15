from pyproj import CRS
from pyproj import Transformer

from shapely.ops import transform as shapely_transform


def transform(src_epsg, dest_epsg, shapely_polygons):
    """
    Transforms a shapely polygon from source to dest epsg
    :param src_epsg:
    :param dest_epsg:
    :param shapely_polygon:
    :return:

    src_epsg = CRS.from_epsg("25830")
    dest_epsg = CRS.from_epsg("4258")

    poly = {'type': 'Polygon',
            'coordinates': [[(443082.0, 4622300.0),
                             (443082.0, 4622288.0),
                             (443094.0, 4622288.0),
                             (443094.0, 4622300.0),
                             (443082.0, 4622300.0)]]}

    transform_poly(src_epsg, dest_epsg, poly)
    """
    return_list = isinstance(shapely_polygons, list)
    if not return_list:
        shapely_polygons = [shapely_polygons]

    src_epsg = CRS.from_user_input(src_epsg)
    dest_epsg = CRS.from_user_input(dest_epsg)
    if src_epsg == dest_epsg:
        return shapely_polygons if return_list else shapely_polygons[0]

    lst = []
    for polygon in shapely_polygons:
        transformer = Transformer.from_crs(src_epsg, dest_epsg, always_xy=True)
        lst.append(shapely_transform(transformer.transform, polygon))

    return lst if return_list else lst[0]
