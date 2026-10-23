"""Unbounded Poisson feed; trajectory count controls Monte Carlo accuracy only."""
import numpy as np


def concentration_only_change(before, after):
    """Whether dilute transport responses remain valid for the new inputs."""
    from copy import deepcopy
    left = deepcopy(before.to_dict() if hasattr(before, "to_dict") else before)
    right = deepcopy(after.to_dict() if hasattr(after, "to_dict") else after)
    left["cell"].pop("concentration_per_ml", None)
    right["cell"].pop("concentration_per_ml", None)
    for section in (left,right):
        section['cell'].setdefault('brownian_enabled',None)
        for key in ('workers','particle_backend'):section['numerics'].pop(key,None)
    return left == right


class ContinuousCellStream:
    """Extend observation time without restarting the source or recycling IDs."""
    def __init__(self, rate, delays, exited, seed=43, start_time=0.):
        self.rate = float(rate)
        self.delays = np.asarray(delays, float)
        self.exited = np.asarray(exited, bool)
        if self.rate < 0 or not len(self.delays) or self.delays.shape != self.exited.shape:
            raise ValueError("Invalid continuous source")
        self.rng = np.random.default_rng(seed)
        self.next_time = start_time+(self.rng.exponential(1/self.rate) if self.rate else np.inf)
        self.injections = []; self.templates = []

    def extend(self, horizon):
        while self.next_time <= horizon:
            self.injections.append(self.next_time)
            self.templates.append(int(self.rng.integers(len(self.delays))))
            self.next_time += self.rng.exponential(1/self.rate)
        arrival = np.asarray(self.injections, float)
        template = np.asarray(self.templates, int)
        outlet = arrival+self.delays[template]
        outlet[~self.exited[template]] = np.nan
        keep = arrival <= horizon
        return {"event_id": np.flatnonzero(keep), "injection_s": arrival[keep],
                "template_id": template[keep], "nozzle_s": outlet[keep]}


def simulate_events(config, particles):
    rate = config.cell.concentration_per_ml*config.fluid.sample_flow_ul_min/1000/60
    valid = particles.residence_time[particles.exited]
    preroll = float(np.max(valid)) if config.events.startup == "steady" and len(valid) else 0.
    stream = ContinuousCellStream(rate, particles.residence_time, particles.exited,
                                  config.numerics.seed+1, -preroll)
    records = stream.extend(max(config.events.durations_s))
    injected, nozzle = records["injection_s"], records["nozzle_s"]
    rng = np.random.default_rng(config.numerics.seed+2)
    eta = config.events.collection_efficiency
    detected = None if eta is None else np.where(rng.random(len(injected))<eta, nozzle, np.nan)
    summaries = []
    for duration in config.events.durations_s:
        exits = np.sort(nozzle[np.isfinite(nozzle) & (nozzle>=0) & (nozzle<=duration)])
        gaps = np.diff(exits)
        bins = np.r_[np.arange(0, duration, 1.), duration]
        counts = np.histogram(exits, bins=bins)[0]
        present = (injected<=duration)&(np.isnan(nozzle)|(nozzle>duration))
        unresolved = present & ~particles.exited[records["template_id"]]
        summaries.append({"duration_s": duration, "injected": int(np.sum((injected>=0)&(injected<=duration))),
            "initial_in_pipe": int(np.sum((injected<0)&(np.isnan(nozzle)|(nozzle>=0)))),
            "nozzle_exits": len(exits), "observed_nozzle_cells_s":len(exits)/duration,
            "upstream_in_transit": int(np.sum(present&~unresolved)),
            "transport_unresolved": int(np.sum(unresolved)),
            "downstream_state": "crossed middle lip; exterior cell dynamics / MS capture unresolved",
            "detected": None if detected is None else int(np.sum((detected>=0)&(detected<=duration))),
            "one_second_count_cv": float(np.std(counts)/np.mean(counts)) if counts.mean()>0 else None,
            "close_event_fraction": float(np.mean(gaps<config.events.resolving_time_ms/1000)) if len(gaps) else None,
            "startup": config.events.startup})
    steady_rate=rate*float(np.mean(particles.exited))
    delta=config.events.resolving_time_ms*1e-3
    return {"rate_cells_s": rate, "steady_nozzle_cells_s":steady_rate,
            "time_isolated_nozzle_cells_s":steady_rate*np.exp(-2*steady_rate*delta),
            "isolation_definition":"both nearest-neighbor gaps >= resolving_time_ms; independent steady Poisson approximation",
            "ms_entry_cells_s":None,"ms_detected_single_cell_events_s":None,
            "rate_censoring_note":"exit fraction uses finite transport sampling; censored trajectories are not proven permanent losses",
            "collection_efficiency": eta,
            "summaries": summaries, **records,
            "hypothetical_detection_s": detected,
            "trajectory_template_count": len(particles.residence_time), "preroll_s": preroll,
            "source": "continuous Poisson feed, unlimited event count; finite observation window",
            "note": "Templates approximate residence distribution; cells crossing the lip remain counted downstream, not deleted. MS capture unknown."}
