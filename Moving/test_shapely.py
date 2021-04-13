from shapely.ops import unary_union
from shapely.geometry import Polygon, mapping
import numpy as np

rectangles = [[(7, 15), (7, 22), (20, 22), (20, 15)], [(4, 8), (4, 18), (12, 18), (12, 8)],
              [(7, 4), (7, 11), (20, 11), (20, 4)]]

polygons = []

for rectangle in rectangles:
    polygons.append(Polygon(rectangle))

polygons = unary_union(polygons)

points = np.array(mapping(polygons)['coordinates'][0])
lines = []

for i in range(points.shape[0]-1):
    lines.append([points[i], points[i+1]])

lines = np.array(lines)

lines_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]

lines = lines[lines_index]

print(lines)