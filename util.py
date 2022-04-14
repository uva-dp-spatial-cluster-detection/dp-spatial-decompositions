"""Utility functions that don't fit elsewhere.
"""

from shapely import geometry


def intersection_over_union(
    region_a: geometry.Polygon, region_b: geometry.Polygon
) -> float:
    """Computes the intersection-over-union (IoU) of two Shapely regions.

    Args:
        region_a (geometry.Polygon): First region.
        region_b (geometry.Polygon): Second region.

    Returns:
        float: Ratio of (area of A intersect B) / (area of A union B)
    """
    return region_a.intersection(region_b).area / region_a.union(region_b).area


def make_shapely_circle(center, radius):
    return geometry.Point(center).buffer(radius)
