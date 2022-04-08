"""Tools for generating test data.
"""

from turtle import shape
from shapely import geometry
import numpy as np
import typing
from cormode import Point


def generate_points_no_cluster(
    entire_region: geometry.Polygon,
    lambda_background: float,
) -> typing.List[Point]:
    background_area = entire_region.area

    points = []

    count = np.random.poisson(lambda_background * background_area)
    while len(points) < count:
        point_x = np.random.uniform(entire_region.bounds[0], entire_region.bounds[2])
        point_y = np.random.uniform(entire_region.bounds[1], entire_region.bounds[3])

        shapely_point = geometry.Point(point_x, point_y)
        if entire_region.contains(shapely_point):
            points.append(Point(point_x, point_y))

    return points


def generate_points_single_cluster(
    entire_region: geometry.Polygon,
    lambda_background: float,
    cluster_region: geometry.Polygon,
    lambda_cluster: float,
) -> typing.List[Point]:
    """Generates list of points according to a Poisson point process.

    Args:
        entire_region (geometry.Polygon): Shapely polygon corresponding to the entire region under consideration.
        lambda_background (float): Parameter of the Poisson distribution for background case counts per unit area. This will be normalized to reflect the region's area.
        cluster_region (geometry.Polygon): Shapely polygon corresponding to the cluster.
        lambda_cluster (float): Parameter of the Poisson distribution for the cluster. This willl be normalized to reflect the region's area.

    Returns:
        typing.List[Point]: List of generated points.
    """

    # make sure cluster region does not extend out of the entire region
    cluster_region = entire_region.intersection(cluster_region)

    # compute area of (entire region) - (cluster)
    background_area = entire_region.area - cluster_region.area

    points = []

    # how many background points should we generate?
    # we scale the Poisson parameter by the area of the region
    background_count = np.random.poisson(lambda_background * background_area)

    # hack: instead of doing the math to figure out how to uniformly distribute points within this shape,
    # we'll just generate points randomly and toss them out if they happen to be in the cluster region.
    bg_generated = 0
    while bg_generated < background_count:
        point_x = np.random.uniform(entire_region.bounds[0], entire_region.bounds[2])
        point_y = np.random.uniform(entire_region.bounds[1], entire_region.bounds[3])

        shapely_point = geometry.Point(point_x, point_y)

        if entire_region.contains(shapely_point) and not cluster_region.contains(
            shapely_point
        ):
            points.append(Point(point_x, point_y))
            bg_generated += 1

    # how many cluster points should we generate?
    # we scale the Poisson parameter by the area of the region
    cluster_count = np.random.poisson(lambda_cluster * cluster_region.area)

    # same hack here
    cluster_generated = 0
    while cluster_generated < cluster_count:
        point_x = np.random.uniform(cluster_region.bounds[0], cluster_region.bounds[2])
        point_y = np.random.uniform(cluster_region.bounds[1], cluster_region.bounds[3])

        shapely_point = geometry.Point(point_x, point_y)

        if entire_region.contains(shapely_point) and cluster_region.contains(
            shapely_point
        ):
            points.append(Point(point_x, point_y))
            cluster_generated += 1

    return points
