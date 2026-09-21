import unittest

from simulation.isaac_lab.g1_stairs.step_up_state import (
    FootSupportObservation,
    StepUpDetectorConfig,
    StepUpObservation,
    StepUpState,
    StepUpStateDetector,
)


DT = 0.05
FORCE = 100.0


def foot(
    *,
    upper=True,
    height=True,
    force=FORCE,
    speed=0.0,
):
    return FootSupportObservation(
        in_upper_region=upper,
        height_ok=height,
        vertical_force_n=force,
        speed_mps=speed,
    )


def obs(
    *,
    left=None,
    right=None,
    upright=1.0,
    base_past=False,
    base_height=False,
    forbidden=False,
    continuation=0.0,
):
    return StepUpObservation(
        left=left or foot(upper=False, height=False, force=0.0),
        right=right or foot(upper=False, height=False, force=0.0),
        upright_gravity_z=upright,
        base_past_riser=base_past,
        base_height_ok=base_height,
        forbidden_contact=forbidden,
        upper_forward_continuation_m=continuation,
    )


class TestStepUpStateDetector(unittest.TestCase):

    def make_detector(self):
        return StepUpStateDetector(
            StepUpDetectorConfig(
                minimum_support_duration_s=0.15,
                maximum_supported_foot_speed_mps=0.15,
                minimum_upright_gravity_z=0.80,
                minimum_vertical_support_force_n=50.0,
                completion_forward_distance_m=0.40,
            ),
            dt=DT,
        )

    def establish_left_support(self, detector):
        result = None
        for _ in range(3):
            result = detector.update(
                obs(left=foot())
            )
        return result

    def test_brief_contact_is_not_support(self):
        detector = self.make_detector()

        detector.update(obs(left=foot()))
        detector.update(
            obs(
                left=foot(
                    upper=False,
                    height=False,
                    force=0.0,
                )
            )
        )

        self.assertEqual(
            detector.state,
            StepUpState.APPROACH,
        )

    def test_fast_sliding_contact_is_not_support(self):
        detector = self.make_detector()

        for _ in range(10):
            detector.update(
                obs(
                    left=foot(speed=0.5)
                )
            )

        self.assertEqual(
            detector.state,
            StepUpState.APPROACH,
        )

    def test_insufficient_force_is_not_support(self):
        detector = self.make_detector()

        for _ in range(10):
            detector.update(
                obs(
                    left=foot(force=10.0)
                )
            )

        self.assertEqual(
            detector.state,
            StepUpState.APPROACH,
        )

    def test_stable_first_support_requires_dwell(self):
        detector = self.make_detector()

        first = detector.update(obs(left=foot()))
        second = detector.update(obs(left=foot()))
        third = detector.update(obs(left=foot()))

        self.assertFalse(first.entered_first_support)
        self.assertFalse(second.entered_first_support)

        self.assertTrue(third.entered_first_support)
        self.assertEqual(
            third.state,
            StepUpState.FIRST_SUPPORT,
        )

    def test_falling_forward_does_not_count_as_base_transition(self):
        detector = self.make_detector()
        self.establish_left_support(detector)

        result = detector.update(
            obs(
                left=foot(),
                upright=0.30,
                base_past=True,
                base_height=True,
            )
        )

        self.assertEqual(
            result.state,
            StepUpState.FIRST_SUPPORT,
        )
        self.assertFalse(result.entered_base_above)

    def test_forbidden_contact_blocks_progress(self):
        detector = self.make_detector()
        self.establish_left_support(detector)

        result = detector.update(
            obs(
                left=foot(),
                base_past=True,
                base_height=True,
                forbidden=True,
            )
        )

        self.assertEqual(
            result.state,
            StepUpState.FIRST_SUPPORT,
        )

    def test_full_valid_step_sequence(self):
        detector = self.make_detector()

        result = self.establish_left_support(detector)

        self.assertTrue(result.entered_first_support)

        result = detector.update(
            obs(
                left=foot(),
                base_past=True,
                base_height=True,
            )
        )

        self.assertTrue(result.entered_base_above)
        self.assertEqual(
            result.state,
            StepUpState.BASE_ABOVE,
        )

        # Right foot also needs full support dwell.
        for _ in range(2):
            result = detector.update(
                obs(
                    left=foot(),
                    right=foot(),
                    base_past=True,
                    base_height=True,
                )
            )

        self.assertEqual(
            result.state,
            StepUpState.BASE_ABOVE,
        )

        result = detector.update(
            obs(
                left=foot(),
                right=foot(),
                base_past=True,
                base_height=True,
            )
        )

        self.assertTrue(
            result.entered_both_feet_supported
        )

        result = detector.update(
            obs(
                left=foot(),
                right=foot(),
                base_past=True,
                base_height=True,
                continuation=0.40,
            )
        )

        self.assertTrue(result.entered_completed)
        self.assertEqual(
            result.state,
            StepUpState.COMPLETED,
        )

    def test_transition_is_paid_only_once(self):
        detector = self.make_detector()

        first_transition_count = 0

        for _ in range(20):
            result = detector.update(
                obs(left=foot())
            )

            first_transition_count += int(
                result.entered_first_support
            )

        self.assertEqual(first_transition_count, 1)

    def test_reset_clears_state_and_dwell(self):
        detector = self.make_detector()

        self.establish_left_support(detector)

        self.assertEqual(
            detector.state,
            StepUpState.FIRST_SUPPORT,
        )

        detector.reset()

        self.assertEqual(
            detector.state,
            StepUpState.APPROACH,
        )

        result = detector.update(obs(left=foot()))

        self.assertFalse(result.left_supported)
        self.assertEqual(
            result.state,
            StepUpState.APPROACH,
        )

    def test_halfway_reset_does_not_pay_first_support_again(self):
        detector = self.make_detector()

        detector.reset(
            initial_state=StepUpState.FIRST_SUPPORT
        )

        result = detector.update(
            obs(left=foot())
        )

        self.assertFalse(
            result.entered_first_support
        )


if __name__ == "__main__":
    unittest.main()
