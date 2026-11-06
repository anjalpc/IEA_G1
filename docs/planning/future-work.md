# Future work

This document is the single backlog for ideas and capabilities that are not
part of the current verified system contract. Items here are proposals, not
operator instructions or claims about deployed behavior.

## Documentation consolidation

The repository documentation is currently scattered and sometimes repeated
across setup guides, architecture specifications, the README, and the system
contract.

Planned cleanup:

- Add a documentation index with one authoritative owner for each subject.
- Keep `docs/architecture/` limited to designs, boundaries, and specifications.
- Keep `docs/setup/` limited to executable operator procedures.
- Keep `docs/reference/` limited to verified contracts, interfaces, and facts.
- Keep unimplemented proposals and milestones in `docs/planning/`.
- Replace duplicated explanations with links to the authoritative document.
- Check commands in setup guides against the current launch files and scripts.
- Archive or remove obsolete instructions after their replacements are verified.

Completion evidence:

- Every maintained document is reachable from one index.
- Each subject has one authoritative explanation.
- Setup guides contain tested commands without conflicting alternatives.
- Repository searches find no contradictory startup or safety procedures.

## Resilient event connectivity

Investigate a dual-link event network in which the robot can use dedicated
Wi-Fi 6/6E as the primary local connection and 5G as a wider-area backup.

Proposed direction:

- Keep balance, hardware control, safety, localization, obstacle avoidance,
  and time-critical navigation onboard.
- Use Tailscale for encrypted addressing, administration, and network roaming.
- Evaluate Zenoh bridges for selected ROS 2 topics, services, and actions
  instead of forwarding unrestricted DDS traffic over the wide-area link.
- Keep raw camera and LiDAR streams local unless a measured use case requires
  remote transport.
- Require stale-command timeouts and a controlled local stop when the remote
  operator heartbeat is lost.
- Retain an independent physical emergency-stop mechanism.
- Treat the laptop as an operator and monitoring station, not as a dependency
  for immediate balance or collision response.

Required validation before event use:

- Measure direct and relayed Tailscale latency, jitter, throughput, and loss.
- Test Wi-Fi-to-5G and 5G-to-Wi-Fi transitions while both links are available.
- Inject packet loss, delay, reordering, and complete disconnection.
- Verify that stale remote commands never continue driving the robot.
- Verify controlled stopping and autonomous recovery after link loss.
- Test inside a crowded venue using the intended modem, SIMs, router, and
  antennas.

## Onboard navigation deployment

The present physical-navigation workflow splits sensor publication on PC2 and
navigation processing on the development laptop. Investigate moving the
headless navigation stack to suitable onboard companion compute.

Planned evaluation:

- Measure PC2 CPU, memory, GPU, temperature, and power headroom under the full
  interaction and sensor workload.
- Profile GLIM, localization, costmaps, planning, perception, and logging.
- Compare the current CPU GLIM configuration with a validated accelerated
  backend where supported.
- Select onboard hardware only after workload and thermal measurements exist.
- Preserve a laptop workflow for visualization, diagnostics, deployment, and
  supervised fallback testing.

Completion evidence:

- The robot can localize, plan, avoid obstacles, and stop safely without the
  operator laptop.
- Loss of the monitoring connection does not compromise local safety.
- Startup, shutdown, recovery, and resource limits are documented and tested.
