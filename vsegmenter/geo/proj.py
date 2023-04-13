from pyproj import CRS, itransform
from pyproj import Transformer


def transform(src_epsg, dest_epsg, polygon):
    """
    Transforms a shapely polygon from source to dest epsg
    :param src_epsg:
    :param dest_epsg:
    :param polygon:
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
    new_poly = polygon.copy()
    rings = polygon["coordinates"]
    tr_rings = []
    transformer = Transformer.from_crs(src_epsg, dest_epsg, always_xy=True)
    for ring in rings:
        tr_ring = transformer.itransform(ring)
        tr_rings.append(list(tr_ring))
    new_poly["coordinates"] = tr_rings
    return new_poly

