"""Classical (i.e., non-private) spatial decompositions for serving 2D counting queries.
"""

import typing
from shapely import geometry
from cormode import Rectangle, Point


class QuadTreeNode:
    # TODO: question: does rounding this to an integer produce better results?
    count: float = 0.0
    """Number of points lying inside of this node's rectangle.
    """

    height: int
    """
    Maximum number of edges between this node and a leaf.
    """
    rect: Rectangle
    """
    Rectangle represented by this QuadTreeNode.
    """
    center: Point
    """Centerpoint of this node's rectangle.
    """

    parent: typing.Optional["QuadTreeNode"]
    """Parent node of this QuadTreeNode, if it exists.
    """
    child_ne: typing.Optional["QuadTreeNode"] = None
    """Northeastern child of this QuadTreeNode, if it exists.
    """
    child_nw: typing.Optional["QuadTreeNode"] = None
    """Northwestern child of this QuadTreeNode, if it exists.
    """
    child_se: typing.Optional["QuadTreeNode"] = None
    """Southeastern child of this QuadTreeNode, if it exists.
    """
    child_sw: typing.Optional["QuadTreeNode"] = None
    """Southwestern child of this QuadTreeNode, if it exists.
    """

    epsilon: typing.Optional[float] = None
    """Epsilon associated with this level of the tree, if this node has been
    privatized. Used to reduce variance in query responses.
    """

    def __init__(
        self,
        rect: Rectangle = Rectangle(0, 0, 1, 1),
        height: int = 0,
        parent: "QuadTreeNode" = None,
    ):
        """Recursively builds a quadtree of a given height.

        Args:
            rect (Rectangle): Rectangle representing the bounds of this QuadTreeNode. Defaults to Rectangle(0, 0, 1, 1).
            height (int, optional): Number of edges from node to any given leaf. Defaults to 0.
        """

        self.rect = rect
        self.center = Point((rect.xmin + rect.xmax) / 2, (rect.ymin + rect.ymax) / 2)
        self.height = height
        self.parent = parent

        if height > 0:
            self.child_ne = QuadTreeNode(
                rect=Rectangle(
                    self.center.x,
                    self.center.y,
                    self.rect.xmax,
                    self.rect.ymax,
                ),
                height=height - 1,
                parent=self,
            )
            self.child_se = QuadTreeNode(
                rect=Rectangle(
                    self.center.x,
                    self.rect.ymin,
                    self.rect.xmax,
                    self.center.y,
                ),
                height=height - 1,
                parent=self,
            )
            self.child_nw = QuadTreeNode(
                rect=Rectangle(
                    self.rect.xmin,
                    self.center.y,
                    self.center.x,
                    self.rect.ymax,
                ),
                height=height - 1,
                parent=self,
            )
            self.child_sw = QuadTreeNode(
                rect=Rectangle(
                    self.rect.xmin,
                    self.rect.ymin,
                    self.center.x,
                    self.center.y,
                ),
                height=height - 1,
                parent=self,
            )

    @property
    def is_leaf(self) -> bool:
        """Returns whether this QuadTreeNode is a leaf."""
        # if any of its children are none, this is a leaf
        return self.child_ne is None

    @property
    def children(self) -> typing.Optional[typing.List["QuadTreeNode"]]:
        if self.is_leaf:
            return []
        else:
            return [self.child_ne, self.child_nw, self.child_sw, self.child_se]

    def insert_point(self, point: Point):
        """Inserts a point into this QuadTree, incrementing the counts of each rectangle it falls into from node to leaf.

        Args:
            point (Point): Point to insert.
        """
        assert (self.rect.xmin <= point.x <= self.rect.xmax) and (
            self.rect.ymin <= point.y <= self.rect.ymax
        ), "Point is not in bounds"

        # increment this count by 1
        self.count += 1

        if self.is_leaf:
            # leaves do not need to recurse on children
            return

        # is this point east of center vertical line?
        east = point.x >= self.center.x
        # is this point north of center horizontal line?
        north = point.y >= self.center.y

        # recurse on whichever child contains this point
        if north and east:
            self.child_ne.insert_point(point)
        elif north:
            self.child_nw.insert_point(point)
        elif east:
            self.child_se.insert_point(point)
        else:
            self.child_sw.insert_point(point)

    def count_inside(self, region: geometry.Polygon) -> float:
        """Estimates the number of points inside this QuadTreeNode contained within a given region.

        Args:
            region (geometry.Polygon): Region to count points within.

        Returns:
            float: Estimated number of this QuadTreeNode's points inside region.
        """
        # represent this region's rectangle as a Shapely box.
        box = geometry.box(*self.rect)

        # is this box completely contained within the region?
        if region.contains(box):
            # return this entire count
            return self.count
        else:
            # if not, either:
            # - use uniformity assumption to estimate on leaf
            # - recurse on children
            if self.is_leaf:
                # leaf: cannot recurse deeper
                # use uniformity assumption to estimate the count contained in the intersection
                intersection_area = region.intersection(box).area
                area_fraction = intersection_area / box.area

                return area_fraction * self.count
            else:
                return sum(child.count_inside(region) for child in self.children)
