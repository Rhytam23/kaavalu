# Two-minute pitch

Fill every bracket from `results/`. Never round in our favour.

## Crowd version (pedestrian clip)

**Problem (0:00).** In 2025, India had 8 major stampedes and 129 deaths: temples, a railway
station, a rally and a stadium gate here in Bengaluru. They happen at choke points, where the
people at the gate can't see how full the space behind them has become.

**How counting helps (0:30).** If a cheap box at the gate counts people every second and compares
that with the space's safe capacity, staff get a clear signal to slow or stop entry earlier. No
cloud, no expensive computer, no identities: only counts.

**What we built (0:55).** An approved detector running on ExecuTorch on this Raspberry Pi,
quantised to INT8. [Show the organiser clip output.] It counts people in every frame and turns
occupancy into green, amber or red.

**Proof (1:20).** We hand-counted the same frames. Our error is [X] per frame; tiling the image
with the same model cut it to [Y]. It runs at [Z] ms per frame, [N]× faster than plain PyTorch,
using about [E] millijoules per frame.

**Practical application (1:45).** Gates, platforms, foot-overbridges, temple queues: an
early-warning counter for about the price of a phone.

## Traffic version (vehicle clip)

Same structure. **Problem:** congested junctions and school-zone crossings in Indian cities,
where traffic studies are done by people with clickers a few days a year. **How counting helps:**
continuous, cheap, offline counts per vehicle type and a congestion level per junction. Proof and
application slots stay the same.

## Where each bracket comes from

| Bracket | File | Field |
| --- | --- | --- |
| [X] | `results/evaluation.json` | `runs.<full-frame run>.mae` |
| [Y] | `results/evaluation.json` | `runs.<tiled run>.mae` |
| [Z] | `results/run_summary.json` | `avg_total_ms_per_frame` |
| [N] | `results/benchmark_report.json` | `speedup_vs_torch` of the chosen config |
| [E] | `results/benchmark_report.json` | `energy_mj_per_frame` of the chosen config |

## Words to avoid

"Surveillance", "real-time monitoring system", "would have prevented the stampede",
"100% accurate", and any blame for a specific incident.
