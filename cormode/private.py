"""Private implementations of spatial decompositions.
"""

from cormode.classical import QuadTreeNode
from cormode import dp
import enum


class BudgetStrategy(enum.Enum):
    """Budget strategy for values of epsilon at each level of a quadtree/kd-tree."""

    UNIFORM = "uniform"
    """Uniform budget strategy. The value of epsilon at each level is (epsilon_total / (height + 1)).
    """
    GEOMETRIC = "geometric"
    """Geometric budget strategy, as described in Cormode et al.
    """


def make_private_quadtree(
    nonprivate_root: QuadTreeNode,
    epsilon_total: float,
    budget_strategy: BudgetStrategy = BudgetStrategy.UNIFORM,
) -> QuadTreeNode:
    """Converts a non-private quadtree to a private one by perturbing the counts at each level.

    Args:
        nonprivate_root (QuadTreeNode): Root QuadTreeNode of the original, non-private tree.
        epsilon (float): Privacy parameter.
        budget_strategy (BudgetStrategy, optional): Strategy for distributing our privacy budget across levels of the tree. Defaults to BudgetStrategy.UNIFORM.

    Returns:
        QuadTreeNode: A new QuadTreeNode whose privatized counts are derived from the original QuadTreeNode.

    Notes:
        Certain invariants of classical (non-private) quadtrees do not hold for privatized quadtrees. Such as:
        - a QuadTreeNode's count is equal to the sum of its children's counts.
        - a QuadTreeNode's count is an integer.
        - a QuadTreeNode's count is nonnegative.
    """
    # make new quadtree with same "index" (i.e., rectangle and height)
    private_root = QuadTreeNode(
        rect=nonprivate_root.rect, height=nonprivate_root.height
    )

    height = nonprivate_root.height

    # now, privatizing the tree is just a matter of setting counts based on the nonprivate tree

    def set_counts(nonprivate_node, private_node, depth=0):
        # compute the value of epsilon for this level

        # TODO: test that the budget strategies actually result in root-to-leaf sum of level epsilons <= epsilon_total
        if budget_strategy == BudgetStrategy.UNIFORM:
            # each level's epsilon is equal
            level_epsilon = epsilon_total / (height + 1)
        elif budget_strategy == BudgetStrategy.GEOMETRIC:
            # ugly formula, taken from Cormode et al. pp. 5
            level_epsilon = (
                2 ** ((height - depth) / 3)
                * epsilon_total
                * ((2 ** (1 / 3) - 1) / (2 ** ((height + 1) / 3) - 1))
            )
        else:
            raise NotImplementedError(
                f"Invalid noise budget strategy '{budget_strategy}'"
            )

        # set this node's count based on laplace mechanism
        private_node.count = dp.laplace_mechanism(
            nonprivate_node.count, sensitivity=1, epsilon=level_epsilon
        )

        # keep track of this node's epsilon
        private_node.epsilon = level_epsilon

        # if there are children:
        if not private_node.is_leaf:
            # recurse on each of them
            for children in zip(nonprivate_node.children, private_node.children):
                set_counts(*children, depth=depth + 1)

    set_counts(nonprivate_root, private_root)

    return private_root
