from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

DEFAULT_SIZE = 1000
SeedLike = Union[int, np.random.Generator, None]


def _coerce_rng(rng: SeedLike) -> np.random.Generator:
    """Return a numpy Generator from a seed, Generator, or ``None``."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def add_noise(
    integers: NDArray[np.floating],
    noise_amount: float = 0.1,
    *,
    rng: SeedLike = None,
) -> NDArray[np.floating]:
    """Add zero-mean Gaussian jitter using the supplied random generator."""
    generator = _coerce_rng(rng)
    noise = generator.standard_normal(size=integers.shape[0]) * noise_amount
    return integers + noise


def uniform_quantal_intervals(
    *,
    low: int = 1,
    high: int = 8,
    size: int = DEFAULT_SIZE,
    noise_amount: float = 0.1,
    unit: float = 0.2,
    rng: SeedLike = None,
) -> NDArray[np.floating]:
    """Sample integer multiples of ``unit`` from a uniform integer range.

    Pass a ``rng`` (seed or :class:`numpy.random.Generator`) to make the draw
    deterministic.
    """
    generator = _coerce_rng(rng)
    integers = generator.integers(low, high, size=size)
    noisy = add_noise(integers, noise_amount=noise_amount, rng=generator)
    return unit * noisy


def geometric_quantal_intervals(
    *,
    p: float = 0.5,
    size: int = DEFAULT_SIZE,
    noise_amount: float = 0.1,
    unit: float = 0.2,
    rng: SeedLike = None,
) -> NDArray[np.floating]:
    """Sample geometric counts and map them onto ``unit``-sized bins.

    Provide ``rng`` to obtain repeatable draws.
    """
    generator = _coerce_rng(rng)
    integers = generator.geometric(p=p, size=size)
    noisy = add_noise(integers, noise_amount=noise_amount, rng=generator)
    return unit * noisy


def uniform_intervals(
    *,
    size: int = DEFAULT_SIZE,
    low: float = 0.0,
    high: float = 2.0,
    rng: SeedLike = None,
) -> NDArray[np.floating]:
    """Draw continuous intervals from a uniform distribution.

    Provide ``rng`` to obtain repeatable draws.
    """
    generator = _coerce_rng(rng)
    return generator.uniform(low, high, size=size)


def normal_intervals(
    *,
    size: int = DEFAULT_SIZE,
    loc: float = 1.0,
    scale: float = 1.0,
    rng: SeedLike = None,
) -> NDArray[np.floating]:
    """Draw positive intervals from a truncated normal distribution.

    Provide ``rng`` for reproducible sampling.
    """
    generator = _coerce_rng(rng)
    samples: list[NDArray[np.floating]] = []
    collected = 0
    while collected < size:
        batch = generator.normal(loc=loc, scale=scale, size=(size - collected) * 2 or 2)
        positives = batch[batch > 0]
        if positives.size == 0:
            continue
        samples.append(positives)
        collected += positives.size
    concatenated = np.concatenate(samples)
    return concatenated[:size]


def repeat_template(
    template: Sequence[float] | NDArray[np.floating],
    *,
    size: int = DEFAULT_SIZE,
    noise_amount: float = 0.1,
    rng: SeedLike = None,
) -> NDArray[np.floating]:
    """Tile ``template`` until ``size`` elements are filled, then add noise.

    Provide ``rng`` to make the jitter reproducible.
    """
    base = np.asarray(template, dtype=float)
    if base.size == 0:
        raise ValueError("template must contain at least one value")
    repeats = int(np.ceil(size / base.size))
    intervals = np.tile(base, repeats)[:size]
    return add_noise(intervals, noise_amount=noise_amount, rng=rng)


def clappy_music(
    *,
    size: int = DEFAULT_SIZE,
    noise_amount: float = 0.1,
    unit=0.5,
    rng: SeedLike = None,
) -> NDArray[np.floating]:
    """Approximate Reich's *Clapping Music* pattern with a noisy template."""
    template = [1, 1, 2, 1, 2, 2, 1, 2] * 2 + [3, 3, 2, 4]
    return unit * repeat_template(
        template, size=size, noise_amount=noise_amount, rng=rng
    )
