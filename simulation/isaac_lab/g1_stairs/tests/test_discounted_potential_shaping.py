import torch

from simulation.isaac_lab.g1_stairs.reward_math import (
    discounted_potential_shaping,
)


def test_terminal_sequence_has_zero_discounted_shaping_return():
    gamma = 0.99
    previous = torch.tensor([0.0])
    total = torch.tensor([0.0])

    sequence = (
        (0.10, False),
        (0.20, False),
        (0.15, True),
    )

    for step, (potential, terminal) in enumerate(sequence):
        shaping, previous = discounted_potential_shaping(
            previous,
            torch.tensor([potential]),
            torch.tensor([terminal]),
            discount_gamma=gamma,
        )
        total += (gamma ** step) * shaping

    torch.testing.assert_close(
        total,
        torch.zeros_like(total),
        atol=1.0e-6,
        rtol=0.0,
    )

    torch.testing.assert_close(
        previous,
        torch.zeros_like(previous),
    )


def test_descent_has_negative_shaping():
    shaping, next_potential = discounted_potential_shaping(
        torch.tensor([0.20]),
        torch.tensor([0.10]),
        torch.tensor([False]),
        discount_gamma=0.99,
    )

    assert shaping.item() < 0.0

    torch.testing.assert_close(
        next_potential,
        torch.tensor([0.10]),
    )


def test_lift_drop_cycle_has_zero_discounted_return():
    gamma = 0.99

    up, previous = discounted_potential_shaping(
        torch.tensor([0.0]),
        torch.tensor([0.20]),
        torch.tensor([False]),
        discount_gamma=gamma,
    )

    down, previous = discounted_potential_shaping(
        previous,
        torch.tensor([0.0]),
        torch.tensor([False]),
        discount_gamma=gamma,
    )

    total = up + gamma * down

    torch.testing.assert_close(
        total,
        torch.zeros_like(total),
        atol=1.0e-6,
        rtol=0.0,
    )
