from shapely.ops import unary_union
from shapely.geometry import Polygon, mapping

rectangles = [[(7, 15), (7, 22), (20, 22), (20, 15)], [(4, 8), (4, 18), (12, 18), (12, 8)],
              [(7, 4), (7, 11), (20, 11), (20, 4)]]

polygons = []

for rectangle in rectangles:
    polygons.append(Polygon(rectangle))

polygons = unary_union(polygons)

print(mapping(polygons)['coordinates'][0])