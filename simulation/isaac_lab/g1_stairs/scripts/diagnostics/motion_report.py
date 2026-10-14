"""Offline pre-reset trace audit. No simulator, no policy/reward changes.

Usage: python3 motion_report.py EVALUATION_DIRECTORY
Writes motion_episodes.csv, motion_swings.csv, motion_summary.json (no overwrite).
Unfinished swings remain in the report; a short failed trajectory is not discarded.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(root):
    episodes, swings = [], []
    paths = sorted(root.rglob('trace.jsonl'))
    if not paths:
        raise ValueError('No trace.jsonl files found')
    for path in paths:
        meta = json.loads(path.with_name('trace_metadata.json').read_text())
        with path.open() as stream:
            rows = [json.loads(x) for x in stream if x.strip()]
        if not rows or any('motion_geometry' not in r for r in rows):
            raise ValueError(f'{path}: requires the C4 diagnostic trace patch; rerun tracing')
        if any(b['episode_step'] <= a['episode_step'] for a,b in zip(rows, rows[1:])):
            raise ValueError(f'{path}: multiple episodes/reset jumps; evaluate one episode per env')
        if not (rows[-1].get('terminated') or rows[-1].get('time_out')):
            raise ValueError(f'{path}: incomplete trace')
        episode_file = path.with_name('episodes.csv')
        with episode_file.open() as stream:
            outcomes = list(csv.DictReader(stream))
        if len(outcomes) != 1:
            raise ValueError(f'{path}: expected exactly one episode row')
        outcome = outcomes[0]['outcome']
        foot_ids = [meta['contact_body_names'].index(n) for n in meta['foot_names']]
        event_counts = {str(i): 0 for i in range(1,6)}
        starts, last_loaded = [None, None], [None, None]
        dt = meta['step_dt_s']
        peak_recoil, peak_slip, tracking, peak_accel = 0., 0., [], 0.
        for i,r in enumerate(rows):
            code = r.get('preferred_transition_event_code') or 0
            if code:
                event_counts[str(code)] += 1
            peak_recoil = max(peak_recoil, -r['root_lin_vel_w'][0])
            tracking.extend(abs(a-b) for a,b in zip(r['joint_pos_target_rad'], r['joint_pos_rad']))
            if i:
                peak_accel = max(peak_accel, norm([(a-b)/dt for a,b in zip(r['root_lin_vel_w'], rows[i-1]['root_lin_vel_w'])]))
            for foot in range(2):
                force = r['contact_net_forces_w'][foot_ids[foot]][2]
                if force >= 30:
                    peak_slip = max(peak_slip, r['motion_geometry']['surface_slip_mps'][foot])
                if starts[foot] is None and force <= 15:
                    starts[foot] = last_loaded[foot] if last_loaded[foot] is not None else 0
                terminal = i == len(rows)-1
                if starts[foot] is not None and (force >= 30 or terminal):
                    start = starts[foot]
                    window = rows[start:i+1]
                    points = [v['motion_geometry']['sole_centre_local'][foot] for v in window]
                    length = sum(math.dist(a,b) for a,b in zip(points, points[1:]))
                    displacement = math.dist(points[0], points[-1])
                    bounds = r['motion_geometry']['sole_bounds_local'][foot]
                    x = points[-1][0]
                    tread = meta['tread_depth_m']
                    level = math.floor((x-meta['first_riser_x_local_m'])/tread)
                    # Footprint bounds are proxies, not measured contact-patch area.
                    landing_on_stair = 0 <= level < meta['step_count']
                    edge = meta['first_riser_x_local_m']+level*tread
                    edge_margin = min(bounds[0]-edge, edge+tread-bounds[1]) if landing_on_stair else None
                    events = [v.get('preferred_transition_event_code') for v in window if v.get('preferred_transition_event_code')]
                    swings.append(dict(trace=str(path.relative_to(root)), foot=meta['foot_names'][foot],
                        outcome=outcome, start_s=window[0]['episode_time_s'], end_s=r['episode_time_s'],
                        ended_loaded=force >= 30, observed_liftoff=last_loaded[foot] is not None, samples=len(window), path_m=length,
                        straight_distance_m=displacement, path_ratio=length/displacement if displacement > .005 else None,
                        lateral_range_m=max(p[1] for p in points)-min(p[1] for p in points),
                        backward_path_m=sum(max(0,a[0]-b[0]) for a,b in zip(points,points[1:])),
                        apex_local_m=max(p[2] for p in points), landing_edge_margin_m=edge_margin,
                        touchdown_speed_mps=r['motion_geometry']['surface_speed_mps'][foot],
                        touchdown_load_n=force, transition_codes=json.dumps(events)))
                    starts[foot] = None
                if force >= 30:
                    last_loaded[foot] = i
        c4rows = [r['coordinated_motion'] for r in rows if r.get('coordinated_motion', {}).get('reference_active')]
        rms = math.sqrt(sum(sum(x*x for x in r['swing_error_xyz_m']) for r in c4rows)/(3*len(c4rows))) if c4rows else None
        episodes.append(dict(trace=str(path.relative_to(root)), outcome=outcome,
            duration_s=rows[-1]['episode_time_s'], direct=event_counts['1'], replant=event_counts['2'],
            wrong=event_counts['3'], unresolved=event_counts['4'], ambiguous=event_counts['5'],
            clean=any(r.get('clean_step_over_event',False) for r in rows),
            peak_backward_speed_mps=peak_recoil, peak_loaded_surface_slip_mps=peak_slip,
            root_lateral_range_m=max(r['root_pos_w'][1] for r in rows)-min(r['root_pos_w'][1] for r in rows),
            peak_root_acceleration_mps2=peak_accel, joint_tracking_mae_rad=sum(tracking)/len(tracking),
            peak_absolute_action=max(abs(a) for r in rows for a in r['action']),
            peak_torque_nm=max(abs(a) for r in rows for a in r['applied_torque_nm']),
            reference_xyz_rmse_m=rms, final_dwell_s=rows[-1].get('finish_dwell_s'),
            final_arm_rms_rad=math.sqrt(sum((rows[-1]['joint_pos_rad'][i]-meta['default_joint_pos_rad'][i])**2
                for i,name in enumerate(meta['joint_names']) if any(s in name for s in ('shoulder','elbow','wrist')))/14)))
    outputs = [root/'motion_episodes.csv', root/'motion_swings.csv', root/'motion_summary.json']
    if any(p.exists() for p in outputs):
        raise FileExistsError('Motion report exists; preserve it or choose a fresh evaluation directory')
    write_csv(outputs[0], episodes)
    write_csv(outputs[1], swings)
    summary = {'episodes': len(episodes), 'successes': sum(e['outcome']=='success' for e in episodes),
               'clean_episodes': sum(e['clean'] for e in episodes), 'swing_windows':len(swings),
               'unfinished_swings':sum(not s['ended_loaded'] for s in swings),
               'notes':['Sole envelope is not a measured contact patch.',
                        'Net body force does not identify the specific contacting obstacle.',
                        'Control-rate sampling can miss substep impacts.',
                        'Path ratio is omitted for displacement <= 5 mm.']}
    with outputs[2].open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation_directory', type=Path)
    summarize(parser.parse_args().evaluation_directory)
