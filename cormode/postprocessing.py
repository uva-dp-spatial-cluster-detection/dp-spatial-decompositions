"""Postprocessing functions for improving query accuracy.
"""

import typing
from cormode.classical import QuadTreeNode
from dataclasses import dataclass


_QUADTREE_FANOUT = 4  # value of fanout for a quadtree


@dataclass
class _WrappedQuadTreeNode:
    """Wrapper around a QuadTreeNode used in implementation of build_ols_treee."""

    wrapped_node: QuadTreeNode

    parent: typing.Optional["_WrappedQuadTreeNode"]
    child_ne: typing.Optional["_WrappedQuadTreeNode"]
    child_nw: typing.Optional["_WrappedQuadTreeNode"]
    child_sw: typing.Optional["_WrappedQuadTreeNode"]
    child_se: typing.Optional["_WrappedQuadTreeNode"]

    # data we are holding onto for intermediate computations
    z_value: float = 0  # value of Z_v for this node, initially set to zero

    @property
    def is_leaf(self) -> bool:
        return self.child_ne is None

    @property
    def children(self):
        if self.child_ne is None:
            return []
        else:
            return [self.child_ne, self.child_nw, self.child_sw, self.child_se]


def _wrap_quadtree(
    tree: QuadTreeNode, parent: _WrappedQuadTreeNode = None
) -> _WrappedQuadTreeNode:
    node = _WrappedQuadTreeNode(
        wrapped_node=tree,
        parent=parent,
        child_ne=None,
        child_se=None,
        child_nw=None,
        child_sw=None,
    )

    if not tree.is_leaf:
        node.child_ne = _wrap_quadtree(tree.child_ne, parent=node)
        node.child_nw = _wrap_quadtree(tree.child_nw, parent=node)
        node.child_sw = _wrap_quadtree(tree.child_sw, parent=node)
        node.child_se = _wrap_quadtree(tree.child_se, parent=node)

    return node


def build_ols_tree(tree: QuadTreeNode) -> QuadTreeNode:
    assert (
        tree.epsilon is not None
    ), "Can only build OLS estimator tree from privatized QuadTree."

    # phase 0: precompute "E" arraay

    # find an arbitrary leaf by traversing to bottom
    leaf = tree
    while not leaf.is_leaf:
        leaf = leaf.child_ne

    # traverse from leaf up to root to compute "E" array
    node = leaf  # start at leaf
    e_array = [0]  # start with initial value of zero to simplify implementation
    height = 0  # leaves have height of zero
    while node:
        # from Lemma 4, pp.6 of Cormode et al.
        next_e = e_array[-1] + _QUADTREE_FANOUT**height + node.epsilon**2
        e_array.append(next_e)

        node = node.parent
        height += 1

    # remove leading zero
    e_array = e_array[1:]

    # sanity check: one entry in E[] per level in tree
    assert len(e_array) == tree.height + 1

    # wrap quadtree to store intermediate values
    wrapped_tree = _wrap_quadtree(tree)

    # Phase I: top-down traversal
    # compute values of alpha from top down

    # list of leaf nodes
    leaves: typing.List[_WrappedQuadTreeNode] = []

    def compute_leaves_z_value(node: _WrappedQuadTreeNode, parent_alpha: float = 0):
        alpha = (
            parent_alpha + (node.wrapped_node.epsilon**2) * node.wrapped_node.count
        )
        if node.is_leaf:
            node.z_value = alpha
            leaves.append(node)  # save list of leaves for next step
        else:
            for child in node.children:
                compute_leaves_z_value(child, parent_alpha=alpha)

    compute_leaves_z_value(wrapped_tree)

    # Phase II: bottom-up traversal
    # performs a reverse level-order traversal
    # each node adds its Z value to its parent's Z value

    traversal_queue = leaves
    while traversal_queue:
        # get a node
        next_node = traversal_queue.pop(0)

        if next_node.parent is None:
            break

        # add its z value to the parents' z value
        next_node.parent.z_value += next_node.z_value

        # make sure we only enqueue the parent once for each child
        # achieve this by only letting NE child enqueue parent
        if next_node == next_node.parent.child_ne:
            traversal_queue.append(next_node.parent)

    # Phase III: top-down traversal

    # in this step, we create the final tree
    # create empty clone of the input tree
    ols_tree = QuadTreeNode(rect=tree.rect, height=tree.height)

    def compute_beta(
        ols_tree_node: QuadTreeNode, wrapped_tree_node: _WrappedQuadTreeNode, node_F=0
    ):
        # node's height is the number of edges from it to leaf
        # so its level is its height plus one
        node_level = ols_tree_node.height + 1

        # copy over epsilon values
        ols_tree_node.epsilon = wrapped_tree_node.wrapped_node.epsilon
        ols_tree_node.count = (
            wrapped_tree_node.z_value - _QUADTREE_FANOUT**node_level * node_F
        ) / e_array[node_level - 1]

        for ols_child, wrapped_child in zip(
            ols_tree_node.children, wrapped_tree_node.children
        ):
            compute_beta(
                ols_child,
                wrapped_child,
                node_F=node_F + ols_tree_node.count * ols_tree_node.epsilon**2,
            )

    compute_beta(ols_tree, wrapped_tree)
    return ols_tree
