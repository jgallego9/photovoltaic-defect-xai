"""Balance multilabel image exposure under a per-original cap."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from torch.utils.data import Sampler

from . import CLASS_NAMES, SPLIT_SEED

TOLERANCE = 0.1
MAX_OCCURRENCES = 3


@dataclass(frozen=True)
class BalancePlan:
    patterns: tuple[tuple[int, ...], ...]
    original_counts: tuple[int, ...]
    allocations: tuple[int, ...]
    healthy_count: int
    signature: str

    def to_dict(self) -> dict[str, Any]:
        """Persist only inputs, allocations and their link to ordered originals."""
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BalancePlan:
        return cls(
            tuple(tuple(pattern) for pattern in payload["patterns"]),
            tuple(payload["original_counts"]),
            tuple(payload["allocations"]),
            payload["healthy_count"],
            payload["signature"],
        )

    def support_counts(self) -> tuple[np.ndarray, np.ndarray]:
        patterns = np.asarray(self.patterns).T
        return patterns @ self.original_counts, patterns @ self.allocations


def _validated_input(labels: Any, ids: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(labels, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(CLASS_NAMES):
        raise ValueError("labels must have nonempty shape (N, 6)")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("labels must be binary")
    stable_ids = tuple(ids)
    if len(stable_ids) != len(values) or any(
        not isinstance(value, str) or not value for value in ids
    ):
        raise ValueError("ids must contain one nonempty stable ID per original")
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("Duplicate original IDs are not allowed")
    return values.astype(np.int64), stable_ids


def _signature(labels: np.ndarray, ids: tuple[str, ...], names: tuple[str, ...]) -> str:
    payload = {"class_names": names, "ids": ids, "labels": labels.tolist()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _inventory(labels: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    counts = Counter(tuple(int(value) for value in row) for row in labels if row.any())
    patterns = tuple(sorted(counts))
    return patterns, np.asarray([counts[pattern] for pattern in patterns], dtype=np.int64)


def _band(median: float) -> tuple[int, int]:
    # Decimal prevents e.g. ceil(0.9 * support) from growing through float noise.
    center, margin = Decimal(str(median)), Decimal(str(TOLERANCE))
    lower = int((center * (1 - margin)).to_integral_value(rounding=ROUND_CEILING))
    upper = int((center * (1 + margin)).to_integral_value(rounding=ROUND_FLOOR))
    return lower, upper


def _solve_patterns(
    patterns: np.ndarray,
    counts: np.ndarray,
    lower: int,
    upper: int,
    cap: int,
) -> np.ndarray:
    """Minimize N-scaled L1 distortion, then exactly lexicographically minimize x."""
    size, total = len(counts), int(counts.sum())
    # z = [x_1...x_P, d_1...d_P], d >= |N*x - n*sum(x)|.
    deviations = total * np.eye(size) - counts[:, None] * np.ones((1, size))
    matrix = np.vstack(
        (
            np.hstack((patterns.T, np.zeros((patterns.shape[1], size)))),
            np.hstack((deviations, -np.eye(size))),
            np.hstack((-deviations, -np.eye(size))),
        )
    )
    lb = np.r_[np.full(patterns.shape[1], lower), np.full(2 * size, -np.inf)]
    ub = np.r_[np.full(patterns.shape[1], upper), np.zeros(2 * size)]
    bounds_lower = np.r_[np.ones(size), np.zeros(size)]
    bounds_upper = np.r_[cap * counts, np.full(size, np.inf)]
    objective = np.r_[np.zeros(size), np.ones(size)]

    def solve(
        cost: np.ndarray,
        constraints: LinearConstraint,
        *,
        lexicographic: bool = False,
    ) -> np.ndarray:
        result = milp(
            cost,
            integrality=np.ones(2 * size),
            bounds=Bounds(bounds_lower, bounds_upper),
            constraints=constraints,
            options={"presolve": True, "mip_rel_gap": 0.0},
        )
        if result.status == 2 and not lexicographic:
            raise ValueError("No balance satisfies the band and occurrence cap")
        if result.status != 0 or result.x is None:
            raise RuntimeError(f"MILP failed: {result.message}")
        rounded = np.rint(result.x)
        if not np.allclose(result.x, rounded, rtol=0, atol=1e-5):
            raise RuntimeError("MILP returned noninteger allocations")
        return rounded.astype(np.int64)

    solution = solve(objective, LinearConstraint(matrix, lb, ub))
    first_x = solution[:size]
    optimum = sum(
        abs(total * int(x) - int(n) * int(first_x.sum())) for x, n in zip(first_x, counts)
    )
    if int(solution[size:].sum()) != optimum:
        raise RuntimeError("The scaled L1 objective is not exact at the optimum")
    # A sum of integer deviations fixes the first objective without epsilon guards.
    matrix = np.vstack((matrix, objective))
    lb, ub = np.r_[lb, optimum], np.r_[ub, optimum]
    for index in range(size):
        lex_cost = np.zeros(2 * size)
        lex_cost[index] = 1
        solution = solve(lex_cost, LinearConstraint(matrix, lb, ub), lexicographic=True)
        bounds_lower[index] = bounds_upper[index] = solution[index]
    allocated = solution[:size]
    exact = sum(
        abs(total * int(x) - int(n) * int(allocated.sum())) for x, n in zip(allocated, counts)
    )
    supports = patterns.T @ allocated
    if (
        exact != optimum
        or np.any(allocated < 1)
        or np.any(allocated > cap * counts)
        or np.any(supports < lower)
        or np.any(supports > upper)
    ):
        raise RuntimeError("The final lexicographic solution violates the exact objective/band")
    return allocated


def build_balance_plan(labels: Any, *, ids: Sequence[str]) -> BalancePlan:
    """Balance the six training labels jointly at median ±10%, with at most three views."""
    values, stable_ids = _validated_input(labels, ids)
    names, max_occurrences = CLASS_NAMES, MAX_OCCURRENCES
    pattern_keys, counts = _inventory(values)
    patterns = np.asarray(pattern_keys, dtype=np.int64).reshape(-1, len(names))
    support = values.sum(axis=0)
    median = float(np.median(support))
    lower, upper = _band(median)
    if np.any(support == 0):
        raise ValueError("At least one positive class has no original images")
    if lower > upper:
        raise ValueError("The rounded median band contains no integer")
    minimum, maximum = patterns.sum(axis=0), max_occurrences * support
    if np.any(minimum > upper) or np.any(maximum < lower):
        raise ValueError("A class cannot reach the prescribed support band")
    healthy = tuple(int(index) for index in np.flatnonzero(values.sum(axis=1) == 0))
    healthy_budget = round(median) if healthy else 0
    if healthy_budget > max_occurrences * len(healthy):
        raise ValueError(
            "Healthy budget exceeds the per-original exposure cap",
        )
    allocations = _solve_patterns(patterns, counts, lower, upper, max_occurrences)
    return BalancePlan(
        pattern_keys,
        tuple(int(value) for value in counts),
        tuple(int(value) for value in allocations),
        len(healthy),
        _signature(values, stable_ids, names),
    )


class BalancedPatternSampler(Sampler):
    """Sample balanced epochs while rotating through the originals in each label pattern."""

    def __init__(self, labels: Any, plan: BalancePlan, *, ids: Sequence[str]) -> None:
        values, self.ids = _validated_input(labels, ids)
        names = CLASS_NAMES
        if _signature(values, self.ids, names) != plan.signature:
            raise ValueError("Balance plan does not match these original IDs/labels/order")
        keys, counts = _inventory(values)
        if keys != plan.patterns or tuple(counts) != plan.original_counts:
            raise ValueError("Balance plan pattern inventory is invalid")
        pattern_array = np.asarray(keys, dtype=np.int64)
        allocations = np.asarray(plan.allocations)
        self._healthy_indices = np.flatnonzero(values.sum(axis=1) == 0)
        if (
            allocations.shape != counts.shape
            or not np.isfinite(allocations).all()
            or not np.equal(allocations, np.rint(allocations)).all()
            or plan.healthy_count != len(self._healthy_indices)
        ):
            raise ValueError("Balance plan does not describe the supplied original population")
        median = float(np.median(values.sum(axis=0)))
        lower, upper = _band(median)
        self._healthy_budget = round(median) if len(self._healthy_indices) else 0
        balanced_support = pattern_array.T @ allocations
        if np.any(balanced_support < lower) or np.any(balanced_support > upper):
            raise ValueError("Balance plan violates the prescribed support band")
        self.epoch = 0
        self._groups: list[np.ndarray] = []
        self._budgets = [int(value) for value in plan.allocations]
        for key in plan.patterns:
            indices = np.flatnonzero(np.all(values == key, axis=1))
            self._groups.append(self._permutation(indices, key))
        if self._healthy_budget:
            self._groups.append(self._permutation(self._healthy_indices, (0,) * len(names)))
            self._budgets.append(self._healthy_budget)
        if any(
            budget < 1 or budget > MAX_OCCURRENCES * len(group)
            for group, budget in zip(self._groups, self._budgets)
        ):
            raise ValueError("Balance plan violates a pattern exposure cap")

    def _permutation(self, indices: np.ndarray, pattern: tuple[int, ...]) -> np.ndarray:
        ordered = np.asarray(sorted(indices.tolist(), key=lambda index: self.ids[index]), dtype=int)
        encoded = json.dumps([SPLIT_SEED, pattern], separators=(",", ":")).encode()
        seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
        return np.random.default_rng(seed).permutation(ordered)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        streams = [
            group[(self.epoch * budget + np.arange(budget)) % len(group)]
            for group, budget in zip(self._groups, self._budgets)
        ]
        # Shuffle the next available pattern, preserving order within each stream.
        rng = np.random.default_rng([SPLIT_SEED, self.epoch])
        remaining = np.asarray(self._budgets, dtype=np.int64)
        positions = np.zeros(len(streams), dtype=int)
        output: list[int] = []
        while remaining.sum():
            draw = int(rng.integers(int(remaining.sum())))
            pattern_index = int(np.searchsorted(np.cumsum(remaining), draw, side="right"))
            output.append(int(streams[pattern_index][positions[pattern_index]]))
            positions[pattern_index] += 1
            remaining[pattern_index] -= 1
        return iter(output)

    def __len__(self) -> int:
        return sum(self._budgets)
