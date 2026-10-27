"""ManagerBasedRLEnv variant that restores a verified near-riser snapshot after reset."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from .runtime_snapshot import (
    ensure_geometry_episode_runtime_for_snapshot,
    load_single_env_snapshot_artifact,
    restore_single_env_snapshot_artifact,
)


class NearRiserSnapshotManagerBasedRLEnv(ManagerBasedRLEnv):
    """Restore an opt-in portable near-riser state after Isaac's normal reset path."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._near_riser_snapshot_artifact = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _snapshot_artifact(self):
        path = getattr(self.cfg, "near_riser_snapshot_path", None)
        if not path:
            return None
        if self._near_riser_snapshot_artifact is None:
            self._near_riser_snapshot_artifact = load_single_env_snapshot_artifact(path)
        return self._near_riser_snapshot_artifact

    def step(self, action):
        result = super().step(action)
        if bool(getattr(self.cfg, "footstep_progress_diagnostic", False)):
            from .footstep_live_progress import audit_after_step
            audit_after_step(self)
        return result

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if bool(getattr(self.cfg, "footstep_progress_diagnostic", False)):
            from .footstep_live_progress import reset_coordinator
            reset_coordinator(self, env_ids)
        artifact = self._snapshot_artifact()
        if artifact is not None:
            ensure_geometry_episode_runtime_for_snapshot(self, artifact)
            restore_single_env_snapshot_artifact(self, artifact, env_ids)


class FootstepManagerBasedRLEnv(NearRiserSnapshotManagerBasedRLEnv):
    """Isolated task: contact update after termination, before reward calculation.

    Isaac Lab 2.3.0 ManagerBasedRLEnv.step calls termination_manager.compute()
    after physics and before reward_manager.compute(). Wrap only this boundary;
    do not override or duplicate the upstream physics/control loop.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        original_compute = self.termination_manager.compute

        def compute_and_update(*args, **kw):

            result = original_compute(*args, **kw)

            from .footstep_live_progress import audit_after_step

            try:
                audit_after_step(self)
            except BaseException as exc:
                import traceback
                print("CONTACT_UPDATE_EXCEPTION:", repr(exc), flush=True)
                traceback.print_exc()
                raise

            return result

        self.termination_manager.compute = compute_and_update
