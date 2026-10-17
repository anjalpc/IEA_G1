"""ManagerBasedRLEnv variant that restores a verified near-riser snapshot after reset."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from .runtime_snapshot import load_single_env_snapshot_artifact, restore_single_env_snapshot_artifact


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

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        artifact = self._snapshot_artifact()
        if artifact is not None:
            restore_single_env_snapshot_artifact(self, artifact, env_ids)
