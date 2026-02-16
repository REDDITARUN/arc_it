"""Augmentation strategies for ARC grids.

Combines proven techniques from:
- VARC (geometric + resolution + translation augmentations)
- ARChitects (color permutations with background preservation)
- NVARC (dihedral group augmentations)

Applied consistently to ALL input/output pairs in a task to preserve
the underlying transformation rule.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ─── Geometric Augmentations ────────────────────────────────────────

def rotate_grid(grid: List[List[int]], k: int = 1) -> List[List[int]]:
    """Rotate grid 90*k degrees counter-clockwise."""
    arr = np.array(grid)
    return np.rot90(arr, k=k).tolist()


def flip_grid(grid: List[List[int]], axis: int = 0) -> List[List[int]]:
    """Flip grid along axis (0=vertical/up-down, 1=horizontal/left-right)."""
    arr = np.array(grid)
    return np.flip(arr, axis=axis).tolist()


def transpose_grid(grid: List[List[int]]) -> List[List[int]]:
    """Transpose grid (swap rows and columns)."""
    arr = np.array(grid)
    return arr.T.tolist()


def identity(grid: List[List[int]]) -> List[List[int]]:
    """No-op, return grid unchanged."""
    return [row[:] for row in grid]


def get_geometric_augmentations() -> List[Tuple[str, Callable]]:
    """Return all 8 dihedral group transformations (D4 symmetry).

    These are the 8 symmetries of a square: 4 rotations x 2 reflections.
    Each is a (name, transform_fn) pair.
    """
    return [
        ("identity", identity),
        ("rot90", lambda g: rotate_grid(g, k=1)),
        ("rot180", lambda g: rotate_grid(g, k=2)),
        ("rot270", lambda g: rotate_grid(g, k=3)),
        ("flip_h", lambda g: flip_grid(g, axis=1)),
        ("flip_v", lambda g: flip_grid(g, axis=0)),
        ("transpose", transpose_grid),
        ("transpose_rot90", lambda g: rotate_grid(transpose_grid(g), k=1)),
    ]


def get_inverse_geometric(name: str) -> Callable:
    """Return the inverse transform for a given geometric augmentation."""
    inverses = {
        "identity": identity,
        "rot90": lambda g: rotate_grid(g, k=3),
        "rot180": lambda g: rotate_grid(g, k=2),
        "rot270": lambda g: rotate_grid(g, k=1),
        "flip_h": lambda g: flip_grid(g, axis=1),
        "flip_v": lambda g: flip_grid(g, axis=0),
        "transpose": transpose_grid,
        "transpose_rot90": lambda g: transpose_grid(rotate_grid(g, k=3)),
    }
    return inverses[name]


# ─── Color Permutations ─────────────────────────────────────────────

def permute_colors(
    grid: List[List[int]],
    perm: List[int],
) -> List[List[int]]:
    """Apply a color permutation to a grid.

    Args:
        grid: 2D list of integers (0-9).
        perm: List of 10 integers, where perm[i] is the new color for color i.
              e.g., [0, 3, 1, 2, 4, 5, 6, 7, 8, 9] swaps colors 1<->3, 2->1, 3->2.
    """
    arr = np.array(grid)
    perm_extended = np.array(perm + list(range(10, arr.max() + 1)) if arr.max() >= 10 else perm)
    return perm_extended[arr].tolist()


def random_color_permutation(
    rng: np.random.RandomState,
    keep_background: bool = True,
) -> List[int]:
    """Generate a random permutation of colors 0-9.

    Args:
        rng: Random state for reproducibility.
        keep_background: If True, color 0 stays mapped to 0.
    """
    perm = list(range(10))
    if keep_background:
        # Permute colors 1-9 only
        subset = perm[1:]
        rng.shuffle(subset)
        perm[1:] = subset
    else:
        rng.shuffle(perm)
    return perm


def inverse_color_permutation(perm: List[int]) -> List[int]:
    """Compute the inverse of a color permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


# ─── Task-Level Augmentation ────────────────────────────────────────

def augment_example(
    example: Dict,
    geometric_fn: Callable,
    color_perm: Optional[List[int]] = None,
) -> Dict:
    """Augment a single input/output example.

    Both input and output are transformed identically to preserve
    the underlying transformation rule.
    """
    augmented = {}
    augmented["input"] = geometric_fn(example["input"])
    if "output" in example:
        augmented["output"] = geometric_fn(example["output"])

    if color_perm is not None:
        augmented["input"] = permute_colors(augmented["input"], color_perm)
        if "output" in augmented:
            augmented["output"] = permute_colors(augmented["output"], color_perm)

    return augmented


def augment_task(
    task: Dict,
    geometric_fn: Callable,
    color_perm: Optional[List[int]] = None,
) -> Dict:
    """Augment all examples in a task consistently.

    Args:
        task: Dict with "train" (list of examples) and "test" (list of examples).
        geometric_fn: A geometric transform function.
        color_perm: Optional color permutation (length 10 list).

    Returns:
        New task dict with all examples augmented.
    """
    augmented = {
        "train": [augment_example(ex, geometric_fn, color_perm) for ex in task["train"]],
        "test": [augment_example(ex, geometric_fn, color_perm) for ex in task["test"]],
    }
    return augmented


def generate_all_augmentations(
    task: Dict,
    num_color_perms: int = 10,
    keep_background: bool = True,
    seed: int = 42,
) -> List[Tuple[Dict, Dict]]:
    """Generate all augmented variants of a task.

    Returns a list of (augmented_task, metadata) pairs where metadata
    contains the augmentation parameters for inversion.

    Total variants = 8 geometric x num_color_perms = 80 by default.
    """
    rng = np.random.RandomState(seed)
    geometrics = get_geometric_augmentations()
    results = []

    for geo_name, geo_fn in geometrics:
        for perm_idx in range(num_color_perms):
            if perm_idx == 0:
                color_perm = list(range(10))  # identity perm
            else:
                color_perm = random_color_permutation(rng, keep_background=keep_background)

            aug_task = augment_task(task, geo_fn, color_perm)
            metadata = {
                "geometric": geo_name,
                "color_perm": color_perm,
                "inverse_color_perm": inverse_color_permutation(color_perm),
            }
            results.append((aug_task, metadata))

    return results
