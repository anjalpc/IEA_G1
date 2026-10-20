import torch

from simulation.isaac_lab.g1_stairs.reward_math import (
    discounted_potential_shaping,
    global_stair_target_update,
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


def test_target_change_debits_potential_once():
    gamma = 0.99
    previous = torch.tensor([0.40])

    debit, previous = discounted_potential_shaping(
        previous,
        torch.tensor([0.35]),
        torch.tensor([True]),
        discount_gamma=gamma,
    )
    repeated, previous = discounted_potential_shaping(
        previous,
        torch.tensor([0.0]),
        torch.tensor([False]),
        discount_gamma=gamma,
    )

    torch.testing.assert_close(debit, torch.tensor([-0.40]))
    torch.testing.assert_close(repeated, torch.tensor([0.0]))
    torch.testing.assert_close(previous, torch.tensor([0.0]))


def test_truncation_keeps_nonterminal_potential_for_bootstrap():
    gamma = 0.99
    shaping, next_potential = discounted_potential_shaping(
        torch.tensor([0.40]),
        torch.tensor([0.40]),
        torch.tensor([False]),
        discount_gamma=gamma,
    )

    torch.testing.assert_close(
        shaping, torch.tensor([gamma * 0.40 - 0.40])
    )
    torch.testing.assert_close(next_potential, torch.tensor([0.40]))


def test_qualified_global_acquisition_debits_old_target_once_and_advances_target():
    gamma = 0.99
    target_step = torch.tensor([[0, 0]], dtype=torch.long)
    physical_touchdown = torch.tensor([[True, False]])

    old_target, acquired, next_target = global_stair_target_update(
        target_step,
        physical_touchdown,
        step_count=3,
    )

    assert old_target.item() == 0
    assert acquired.item() is True
    torch.testing.assert_close(next_target, torch.tensor([[1, 1]]))

    previous = torch.tensor([[0.40, 0.25]])
    current = torch.tensor([[0.35, 0.20]])
    target_boundary = physical_touchdown.any(dim=1, keepdim=True).expand_as(current)

    debit, next_potential = discounted_potential_shaping(
        previous,
        current,
        target_boundary,
        discount_gamma=gamma,
    )
    repeated, next_potential = discounted_potential_shaping(
        next_potential,
        torch.zeros_like(current),
        torch.zeros_like(target_boundary),
        discount_gamma=gamma,
    )

    torch.testing.assert_close(debit, -previous)
    torch.testing.assert_close(repeated, torch.zeros_like(repeated))
    torch.testing.assert_close(next_potential, torch.zeros_like(next_potential))


def test_forward_insertion_uses_reversible_potential_when_enabled():
    import ast
    from pathlib import Path

    source_path = Path(__file__).parents[1] / "geometry_acquisition_rewards.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    update = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "update"
    )

    def name_of(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = name_of(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return None

    calls = [
        node
        for node in ast.walk(update)
        if isinstance(node, ast.Call)
        and name_of(node.func) == "discounted_potential_shaping"
        and len(node.args) >= 3
    ]

    assert any(
        name_of(call.args[0]) == "self.best_insertion"
        and name_of(call.args[1]) == "insertion_score"
        and name_of(call.args[2]) == "potential_boundary"
        for call in calls
    ), "forward insertion is not wired to failure-neutral potential accounting"
