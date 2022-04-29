# implementation of Kulldorff's statistic
# based on Neill, Daniel B. Detection of Spatial and Spatio-Temporal Clusters.

import typing
import numpy as np
from cormode import gen, Point
from shapely import geometry
from cormode.classical import QuadTreeNode


def compute_scan_statistic_threshold(
    lambda_background, population, significance_level=0.05, iter=1000
):
    # compute scan statistic assuming background parameters
    scan_statistic_values = []
    for _ in range(iter):
        # generate some background data
        points = gen.generate_points_no_cluster(
            geometry.box(0, 0, 1, 1), lambda_background
        )
        # build quadtree
        tree = QuadTreeNode(height=5)
        for x, y in points:
            tree.insert_point(Point(x, y))
        # compute max scan statistic
        scan_statistic, _ = find_max_kulldorff_sweep(tree, population)
        scan_statistic_values.append(scan_statistic)

    # return (1-significance_level) percentile value
    return np.percentile(scan_statistic_values, 1 - significance_level)


def find_max_kulldorff_sweep(
    root: QuadTreeNode,
    population: float,
    center_xs=np.linspace(0, 1, 10),
    center_ys=np.linspace(0, 1, 10),
    radii=np.linspace(0.1, 0.25, 5),
) -> typing.Tuple[float, typing.Tuple[Point, float]]:
    # total case count
    total_count = root.count

    # area of entire space under consideration
    total_area = geometry.box(*root.rect).area

    max_scan_statistic = None
    max_region = None

    # sweep a circle across the space
    for x in center_xs:
        for y in center_ys:
            for radius in radii:
                # create circular region at (x,y) of radius
                region = geometry.Point(x, y).buffer(radius)
                # estimate the count inside using quadtree
                est_count_inside = root.count_inside(region)
                # TODO: assumption
                # assume population is distributed uniformly to estimate population
                est_population_inside = (region.area / total_area) * population

                scan_statistic = kulldorff_scan_statistic(
                    est_count_inside, est_population_inside, total_count, population
                )

                if max_scan_statistic is None or scan_statistic > max_scan_statistic:
                    max_scan_statistic = scan_statistic
                    max_region = ((x, y), radius)

    return max_scan_statistic, max_region


def kulldorff_scan_statistic(
    c_in: float, b_in: float, c_all: float, b_all: float
) -> float:
    """Computes the log-Kulldorff scan statistic.

    To avoid floating-point overflow, computes the logarithm of the Kulldorff scan statistic.

    Args:
        c_in (float): Case count inside the region.
        b_in (float): Population inside the region.
        c_all (float): Case count in total.
        b_all (float): Population in total.

    Returns:
        float: Value of F, the scan statistic
    """
    c_out = c_all - c_in
    b_out = b_all - b_in

    if (c_in / b_in) <= (c_out / b_out):
        return 0

    return (
        c_in * np.log(c_in / b_in)
        + c_out * np.log(c_out / b_out)
        - c_all * np.log(c_all / b_all)
    )
