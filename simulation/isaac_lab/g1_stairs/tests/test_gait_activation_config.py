"""Configuration-only regression check; run directly with host Python, no Kit."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS


def check():
    # Execute the actual configuration function; replace only Isaac containers.
    path = Path(__file__).parents[1] / 'gait_runtime.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'configure_gait')
    scope = dict(math=math, GaitFrequencyActionCfg=NS, RewardTermCfg=NS,
                 physical_last_action=None, physical_action_rate=None,
                 phase_swing_contact=None)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)

    def configure(options):
        env = NS(actions=NS(), rewards=NS(action_rate=NS()),
                 observations=NS(policy=NS(last_action=NS(history_length=5)),
                                 critic=NS(last_action=NS(history_length=5))))
        resolved = dict(gait=dict(enabled=True, mode='learned', **options),
                        observations=dict(preferred_transition_foot=True,
                                          actor_height_scan=False, critic_height_scan=True),
                        policy_initialization=dict(stair_observations=489, actions=30))
        scope['configure_gait'](env, resolved)
        return env.actions.gait_frequency.first_step_activation_distance_m

    assert configure({}) == .35
    assert configure(dict(first_step_activation_distance_m=.70)) == .70
    for value in (0, -.1, float('nan'), float('inf'), -float('inf')):
        try:
            configure(dict(first_step_activation_distance_m=value))
        except ValueError as error:
            assert 'first_step_activation_distance_m' in str(error)
        else:
            raise AssertionError(f'accepted invalid distance: {value}')
    print('PASS: default, override, and invalid activation distances')


if __name__ == '__main__':
    check()
