"""Implementation of expectation-based Poisson scan statistic, as described
in Neill 2006, "Detection of Spatial and Spatio-Temporal Clusters."
"""
import numpy as np
import typing
from shapely import geometry
from cormode import Point
from cormode.classical import QuadTreeNode


def find_max_ebp_sweep(
    root: QuadTreeNode,
    lambda_background: float,
    center_xs=np.linspace(0, 1, 10),
    center_ys=np.linspace(0, 1, 10),
    radii=np.linspace(0.1, 0.25, 5),
) -> typing.Tuple[float, typing.Tuple[Point, float]]:
    """Finds the region that maximizes the expectation-based Poisson scan statistic.
    Sweeps a circle over the area, computing the scan statistic at each region. Returns
    the (scan statistic, (region center, region radius)) that maximizes the EBP scan
    statistic.

    Args:
        root (QuadTreeNode): Root of the QuadTreeNode containing spatial counts.
        lambda_background (float): Poisson parameter of the background distribution assuming no cluster.
                                    In practice this will be estimated based on prior data.
        center_xs (optional): X coordinates of sweeping circle centers to try. Defaults to np.linspace(0, 1, 10).
        center_ys (_type_, optional): Y coordinates of sweeping circle centers. Defaults to np.linspace(0, 1, 10).
        radii (_type_, optional): Radii of sweeping circles. Defaults to np.linspace(0.1, 0.25, 5).

    Returns:
        typing.Tuple[float, typing.Tuple[Point, float]]: Tuple of (max scan statistic, circle) where circle is a tuple of (center, radius).
    """

    # area of entire space under consideration
    total_area = geometry.box(*root.rect).area

    max_scan_statistic = None
    max_region = None

    # sweep a circle across the space
    for x in center_xs:
        for y in center_ys:
            for radius in radii:
                region = geometry.Point(x, y).buffer(radius)

                # TODO: assumption that baseline is uniform across the entire area
                # ...would need real historical data for this...
                baseline = lambda_background * (region.area / total_area)

                # compute scan statistic for this region
                scan_statistic = ebp_scan_statistic(root.count_inside(region), baseline)

                # if it is a new max, record it
                if max_scan_statistic is None or scan_statistic > max_scan_statistic:
                    max_scan_statistic = scan_statistic
                    max_region = (Point(x, y), radius)

    # return max scan statistic we encountered and its associated region
    return max_scan_statistic, max_region


def ebp_scan_statistic(count_inside, baseline_inside) -> float:
    """Computes the expectation-based Poisson scan statistic for a given region.

    Reference: Neill, 2006. "Detection of Spatial and Spatio-Temporal Clusters."

    Args:
        count_inside (float): Count inside the region.
        baseline_inside (float): Expected count inside of this region, inferred from historical data.

    Returns:
        float: value of EBP scan statistic for this region.
    """

    if count_inside > baseline_inside:
        return (count_inside / baseline_inside) ** count_inside * np.exp(
            baseline_inside - count_inside
        )
    else:
        return 1
