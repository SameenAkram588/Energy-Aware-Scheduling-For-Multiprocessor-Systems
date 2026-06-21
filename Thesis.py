#!/usr/bin/env python3
"""
Mode-switching EDF simulator with SRP (Stack Resource Policy) integrated,
analytic S_L search, and probabilistic LO PMFs.

- SRP: resource ceilings computed as max(preemption_level(task)) for tasks using each resource.
- Jobs execute CSs (from task.cs_map) at the start of the job, non-preemptively.
- If a HI job is blocked by a LO job (resource held), switch to HI mode immediately.

Run: python mode_switch_with_srp.py
Date: 2025-12-07
"""
import random
import itertools
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import time as pytime
import json
import matplotlib.pyplot as plt
import os

# --------------------------
# CONFIG
# --------------------------
RANDOM_SEED = None  # set to int for reproducible runs
TIME_HORIZON = 120.0
QUANTUM = 0.05
NUM_TASKS = 100
RESOURCE_POOL = ["R1", "R2", "R3", "R4", "R5","R6","R7","R8","R9","R10"]

# --------------------------
# MULTICORE CONFIG (ADDED)
# --------------------------
NUM_CORES = 25


# Analytic / PMF globals
RES = 0.1             # PMF discretization resolution (bin width)
MAX_PMF_LEN = 5000    # maximum PMF length for analytic convolution before bailout
TRIM_THRESH = 1e-12

# Candidate speeds and DBF failure threshold
SPEEDS = [round(s,2) for s in np.arange(1.0, 0.0, -0.1)]
F_S_target = 1e-3

# Analytic runtime cap: don't check t beyond MAX_T_CHECK seconds (to avoid huge hyperperiods)
MAX_T_CHECK = 360.0

# PMF / analytic seeding
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

# --------------------------
# Helper: preemption level and SRP ceilings
# --------------------------
def preemption_level_for_task(task):
    # As you supplied: preemption level = 1 / T
    return 1.0 / float(task.period)

def build_resource_ceilings(tasks):
    """Compute ceiling per resource: max preemption_level(task) among tasks that use the resource."""
    ceiling = {}
    for r in RESOURCE_POOL:
        vals = []
        for t in tasks:
            if r in t.resources:
                vals.append(preemption_level_for_task(t))
        if vals:
            ceiling[r] = max(vals)
    return ceiling

# --------------------------
# Task / Job classes
# --------------------------
class Task:
    _name_counter = itertools.count(1)
    def __init__(self, kind, period, c_hi, pmf_lo, resources, cs_map, phase=0.0, name=None,
                 base_C_thr=None, base_C_deg=None):
        self.name = name if name is not None else f"T{next(Task._name_counter)}"
        assert kind in ('HI', 'LO')
        self.kind = kind
        self.period = float(period)
        self.c_hi = float(c_hi)
        self.pmf_lo = pmf_lo[:]  # list of (value, prob)
        total_p = sum(p for (_, p) in self.pmf_lo)
        if total_p > 0:
            self.pmf_lo = [(v, p/total_p) for (v,p) in self.pmf_lo]
        self.resources = list(resources)
        self.cs_map = dict(cs_map)  # resource -> cs length (work units at speed=1)
        self.phase = phase
        self.base_C_thr = base_C_thr if base_C_thr is not None else max(0.5, self.c_hi * 0.6)
        self.base_C_deg = base_C_deg if base_C_deg is not None else max(0.1, min(self.pmf_lo[0][0], self.c_hi) * 0.9)
        self.c_lo = sum(v * p for v, p in self.pmf_lo)
    def sample_lo_execution(self):
        r = random.random()
        cum = 0.0
        for v,p in self.pmf_lo:
            cum += p
            if r <= cum + 1e-12:
                return float(v)
        return float(self.pmf_lo[-1][0])
    def __repr__(self):
        return f"Task({self.name},{self.kind},T={self.period},c_hi={self.c_hi})"

class Job:
    _ids = itertools.count()
    def __init__(self, task: Task, release_time: float, sampled_lo_work=None):
        self.id = next(Job._ids)
        self.task = task
        self.release_time = float(release_time)
        self.deadline = self.release_time + task.period
        if task.kind == 'LO':
            base_work = sampled_lo_work if sampled_lo_work is not None else task.sample_lo_execution()
        else:
            base_work = task.c_hi
        self.base_work = float(base_work)
        # remaining work (excluding CSs) — we will subtract CS durations from remaining when executed
        # For simplicity, treat CS work as part of the remaining work but tracked separately for locking/non-preemptive handling.
        self.remaining_work = float(self.base_work)
        # CS per-resource remaining (work units at speed=1). Copy from task.
        self.cs_remaining = {r: float(self.task.cs_map[r]) for r in self.task.cs_map}
        # order of CS execution: list(resources). We'll execute in this order at job start.
        self.cs_order = list(self.cs_remaining.keys())
        self.executed_time = 0.0   # wall-clock execution time accumulated (seconds)
        self.start_time = None
        self.finish_time = None
        self.dropped = False
        self.blocked = False  # whether currently blocked waiting for resource
        self.block_start_time = None
        self.total_blocking_time = 0.0
        self.total_waiting_time = 0.0   # before first execution
        # To track when CS stage is done; once cs_order emptied we execute normal work.
    def is_complete(self, eps=1e-9):
        return (self.remaining_work <= eps) and (all(v <= eps for v in self.cs_remaining.values()))
    def work_done(self):
        # total work done = base_work - remaining_work; CS consumed reduces remaining_work if we decide so
        # We'll compute actual done as base_work - remaining_work (CSs are included in base_work initially)
        return max(0.0, self.base_work - self.remaining_work)
    def next_pending_cs(self):
        # return next resource to lock (or None)
        for r in self.cs_order:
            if self.cs_remaining.get(r, 0.0) > 1e-12:
                return r
        return None
    def __repr__(self):
        return f"Job({self.task.name}#{self.id},r={self.release_time:.2f},d={self.deadline:.2f},work={self.base_work:.3f})"

# --------------------------
# Random task-set generator
# --------------------------

def generate_random_taskset(n_tasks=10, seed=None):
    if seed is not None:
        rnd = random.Random(seed)
    else:
        rnd = random

    tasks = []

    for i in range(n_tasks):

        kind = rnd.choice(['LO', 'HI'])
        period = rnd.choice([10, 20, 30, 40])

        # Each task can use at MOST one shared resource
        use_resource = rnd.choice([True, False])
        resources = []
        cs_map = {}

        if use_resource:
            r = rnd.choice(RESOURCE_POOL)
            resources = [r]
            cs_map[r] = round(rnd.uniform(0.05, 0.8), 3)

        # Generate per-core execution parameters
        core_c_hi = {}
        core_pmf_lo = {}
        core_c_lo = {}

        for cid in range(NUM_CORES):

            # HI WCET per core
            c_hi = round(rnd.uniform(0.5, 3.0), 3)
            core_c_hi[cid] = c_hi

            # LO PMF per core
            supports = sorted([
                round(rnd.uniform(0.4 * c_hi, c_hi + 0.3), 3)
                for _ in range(4)
            ])

            ps = [rnd.random() for _ in range(4)]
            s = sum(ps)
            ps = [p / s for p in ps]

            pmf_lo = list(zip(supports, ps))
            core_pmf_lo[cid] = pmf_lo

            # Expected LO execution per core
            core_c_lo[cid] = sum(v * p for v, p in pmf_lo)

        # Use core 0 parameters as default (scheduler will override later)
        t = Task(
            kind=kind,
            period=period,
            c_hi=core_c_hi[0],
            pmf_lo=core_pmf_lo[0],
            resources=resources,
            cs_map=cs_map
        )

        # Attach per-core parameters
        t.core_c_hi = core_c_hi
        t.core_pmf_lo = core_pmf_lo
        t.core_c_lo = core_c_lo

        tasks.append(t)

    print("\nGenerated tasks (heterogeneous per-core):")
    for t in tasks:
        print(f" {t.name} crit={t.kind} T={t.period}")
        print(f"   resources: {t.resources}")
        for cid in range(NUM_CORES):
            print(f"   Core {cid}: c_hi={t.core_c_hi[cid]}, "
                  f"E[c_lo]={round(t.core_c_lo[cid],3)}")
            print(f"     PMF_LO:")
            for value, prob in t.core_pmf_lo[cid]:
              print(f"         ({value}, {round(prob, 3)})")

    return tasks


def load_taskset_from_file(filename):
    global NUM_TASKS, RESOURCE_POOL, NUM_CORES

    with open(filename, 'r') as f:
        data = json.load(f)

    # ---- Load constants ----
    NUM_TASKS = data["NUM_TASKS"]
    RESOURCE_POOL = data["RESOURCE_POOL"]
    NUM_CORES = data["NUM_CORES"]

    tasks = []

    # ---- Load tasks ----
    for tdata in data["tasks"]:

        kind = tdata["kind"]
        period = float(tdata["period"])
        resources = tdata["resources"]
        cs_map = tdata["cs_map"]

        core_c_hi = {}
        core_pmf_lo = {}
        core_c_lo = {}

        for cid in range(NUM_CORES):
            cdata = tdata["core_data"][cid]

            c_hi = float(cdata["c_hi"])
            pmf_lo = [(float(v), float(p)) for v, p in cdata["pmf_lo"]]

            core_c_hi[cid] = c_hi
            core_pmf_lo[cid] = pmf_lo
            core_c_lo[cid] = sum(v * p for v, p in pmf_lo)

        # Create task (default = core 0)
        t = Task(
            kind=kind,
            period=period,
            c_hi=core_c_hi[0],
            pmf_lo=core_pmf_lo[0],
            resources=resources,
            cs_map=cs_map,
            name=tdata["name"]
        )

        # Attach per-core data
        t.core_c_hi = core_c_hi
        t.core_pmf_lo = core_pmf_lo
        t.core_c_lo = core_c_lo

        tasks.append(t)

    # ---- Print (same as before) ----
    print("\nLoaded tasks from file:")
    for t in tasks:
        print(f" {t.name} crit={t.kind} T={t.period}")
        print(f"   resources: {t.resources}")
        for cid in range(NUM_CORES):
            print(f"   Core {cid}: c_hi={t.core_c_hi[cid]}, "
                  f"E[c_lo]={round(t.core_c_lo[cid],3)}")
            print("     PMF_LO:")
            for value, prob in t.core_pmf_lo[cid]:
                print(f"         ({value}, {round(prob, 3)})")

    return tasks

def save_taskset_to_file(tasks, filename):
    data = {
        "NUM_TASKS": len(tasks),
        "RESOURCE_POOL": RESOURCE_POOL,
        "NUM_CORES": NUM_CORES,
        "tasks": []
    }

    for t in tasks:
        tdata = {
            "name": t.name,
            "kind": t.kind,
            "period": t.period,
            "resources": t.resources,
            "cs_map": t.cs_map,
            "core_data": []
        }

        for cid in range(NUM_CORES):
            core_entry = {
                "c_hi": t.core_c_hi[cid],
                "pmf_lo": t.core_pmf_lo[cid]
            }
            tdata["core_data"].append(core_entry)

        data["tasks"].append(tdata)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Taskset saved to {filename}")

# --------------------------
# Resource-aware task partition with utilization check
# --------------------------
def partition_tasks_to_cores(tasks, num_cores):

    # ---- Step 1: Build clusters by resource ----
    clusters = defaultdict(list)

    for t in tasks:
        if t.resources:
            clusters[t.resources[0]].append(t)
        else:
            clusters[f"NORES_{t.name}"].append(t)

    # ---- Print clusters (ADDED) ----
    print("\nClusters formed (by resource):")
    for cluster_key, cluster_tasks in clusters.items():
           task_names = [t.name for t in cluster_tasks]
           print(f" {cluster_key} -> {task_names}")

           cores = [[] for _ in range(num_cores)]
           core_util = [0.0 for _ in range(num_cores)]

    # ---- Step 2: Assign each cluster ----
    for cluster_key, cluster_tasks in clusters.items():

        best_core = None
        best_total_exec = float('inf')

        # Try assigning cluster to each core
        for cid in range(num_cores):

            # Compute cluster utilization on this core
            cluster_util = 0.0
            total_exec = 0.0

            for t in cluster_tasks:
                cluster_util += t.core_c_lo[cid] / t.period
                total_exec += t.core_c_lo[cid]

            # Check utilization feasibility
            if core_util[cid] + cluster_util <= 1.0:

                # Choose core where cluster runs fastest
                if total_exec < best_total_exec:
                    best_total_exec = total_exec
                    best_core = cid

        # ---- Fallback if no core satisfies utilization ----
        if best_core is None:
            best_core = min(range(num_cores), key=lambda c: core_util[c])
            print(f"WARNING: Cluster {cluster_key} exceeds utilization, forcing assignment to Core {best_core}")

        # ---- Assign cluster ----
        for t in cluster_tasks:
            cores[best_core].append(t)
            core_util[best_core] += t.core_c_lo[best_core] / t.period

    # Debug print
    print("\nPartition result:")
    for cid in range(num_cores):
        print(f" Core {cid}: {[t.name for t in cores[cid]]} | Utilization={round(core_util[cid],3)}")

    return cores

# --------------------------
# Improved PMF helpers
# --------------------------
def pmf_list_to_array(values, probs, res=RES):
    if len(values) == 0:
        return np.array([1.0])
    max_v = max(values)
    max_bin = int(math.ceil(max_v / res))
    arr = np.zeros(max_bin + 2)
    for v, p in zip(values, probs):
        idxf = v / res
        lo = int(math.floor(idxf))
        hi = lo + 1
        frac_hi = idxf - lo
        frac_lo = 1.0 - frac_hi
        if lo < 0:
            arr[0] += p
        else:
            if lo >= len(arr):
                arr = np.concatenate([arr, np.zeros(lo - len(arr) + 1)])
            arr[lo] += p * frac_lo
        if frac_hi > 1e-15:
            if hi >= len(arr):
                arr = np.concatenate([arr, np.zeros(hi - len(arr) + 1)])
            arr[hi] += p * frac_hi
    s = arr.sum()
    if s > 0.0:
        arr = arr / s
    return arr

def convolve_arrays(a, b):
    c = np.convolve(a, b)
    if len(c) > MAX_PMF_LEN:
        idxs = np.where(c > TRIM_THRESH)[0]
        if len(idxs) > 0:
            c = c[:idxs[-1]+1]
    s = c.sum()
    if s > 0.0:
        c = c / s
    return c

def scale_pmf_array_by_speed(arr, s, res=RES):
    if s <= 0:
        raise ValueError("Speed s must be > 0")
    if arr.sum() == 0:
        return np.array([1.0])
    old_indices = np.nonzero(arr)[0]
    if len(old_indices) == 0:
        return np.array([1.0])
    old_max = old_indices[-1]
    new_max_idx_f = old_max / s
    new_len = int(math.ceil(new_max_idx_f)) + 2
    out = np.zeros(new_len)
    for i, p in enumerate(arr):
        if p <= 0:
            continue
        new_pos = i / s
        j = int(math.floor(new_pos))
        frac = new_pos - j
        if j < 0:
            out[0] += p
        else:
            if j >= len(out):
                out = np.concatenate([out, np.zeros(j - len(out) + 1)])
            out[j] += p * (1.0 - frac)
        if frac > 1e-15:
            j2 = j + 1
            if j2 >= len(out):
                out = np.concatenate([out, np.zeros(j2 - len(out) + 1)])
            out[j2] += p * frac
    if out.sum() > 0.0:
        out = out / out.sum()
    return out

# -----------------------------
# Analytic probabilistic DBF helpers (with safeguards)
# -----------------------------
def lcm(a,b): return abs(a*b) // math.gcd(int(a), int(b))
def hyperperiod(tasks):
    vals = [int(t['T']) for t in tasks]
    hp = vals[0]
    for v in vals[1:]:
        hp = lcm(hp, v)
    return hp

def build_time_check_list(tasks):
    HP = hyperperiod(tasks)
    tset = set()
    for t in tasks:
        kmax = int(HP // t['T'])
        for k in range(1, kmax+1):
            tset.add(round(k * t['T'], 6))
    tlist = sorted(tset)
    return tlist

def compute_prob_exceed_for_t(tasks, s, t_list, res=RES):
    # Precompute single-job LO PMF arrays
    single_arrs = []
    for task in tasks:
        arr = pmf_list_to_array(task['C_LO_vals'], task['C_LO_p'], res=res)
        single_arrs.append(arr)
    results = {}
    for t in t_list:
        per_task_pmfs = []
        for idx, task in enumerate(tasks):
            k = int(math.floor(t / task['T']))
            if k == 0:
                per_task_pmfs.append(np.array([1.0]))
                continue
            single = single_arrs[idx]
            scaled_single = scale_pmf_array_by_speed(single, s, res=res)
            pmf_k = scaled_single.copy()
            for _ in range(k - 1):
                pmf_k = convolve_arrays(pmf_k, scaled_single)
            per_task_pmfs.append(pmf_k)
        # convolve across tasks
        total = per_task_pmfs[0]
        for p in per_task_pmfs[1:]:
            total = convolve_arrays(total, p)
        # threshold index
        idx_th = int(math.floor(t / res))
        if idx_th + 1 < len(total):
            p_exceed = float(np.sum(total[idx_th+1:]))
        else:
            p_exceed = 0.0
        results[t] = p_exceed
    return results

def find_S_L_analytic(analytic_tasks, speeds=None, F_s=F_S_target, res=RES):
    """
    Find S_L by iterating discrete speeds from lowest -> highest and returning
    the first (minimum) speed s for which the analytic probabilistic DBF
    constraint holds: max_t Pr(DBF(t) > t) <= F_s.

    Arguments:
      - analytic_tasks: list of dicts expected by compute_prob_exceed_for_t,
                        each dict has keys 'T', 'C_LO_vals', 'C_LO_p'.
      - speeds: optional list of candidate speeds. If None, uses normalized [0.1,0.2,...,1.0].
      - F_s: target failure threshold (e.g., 1e-3).
      - res: PMF discretization resolution passed to compute_prob_exceed_for_t.
      - max_t_check: maximum t to include (passed to build_time_check_list).

    Returns:
      - selected_s: the chosen S_L (float). If no candidate meets F_s, returns 1.0.
      - per_speed_maxp: dict mapping each tested speed -> max Pr(DBF(t)>t).
    """
    # default normalized discrete speeds from 0.1 to 1.0 (inclusive)
    if speeds is None:
        speeds = [round(x, 2) for x in np.arange(0.1, 1.0 + 1e-9, 0.1)]
    else:
        # ensure sorted ascending (lowest -> highest)
        speeds = sorted(speeds)

    # build the t_list once (cap at max_t_check to avoid huge hyperperiods)
    t_list = build_time_check_list(analytic_tasks)
    print("Time-check list (t):", t_list)
    print("Testing speeds (low -> high):", speeds)

    per_speed_maxp = {}
    for s in speeds:
        print(f"\nTesting candidate speed s = {s:.2f} ...")
        # compute per-t exceedance probabilities using analytic method
        probs = compute_prob_exceed_for_t(analytic_tasks, s, t_list, res=res)
        max_p = max(probs.values()) if probs else 0.0
        per_speed_maxp[s] = max_p
        print(f"  max Pr(DBF(t)>t) across checked t = {max_p:.6e}")
        # if meets probabilistic constraint, return it (minimum s)
        if max_p <= F_s:
            print(f"Selected S_L = {s:.2f} (first speed meeting probabilistic constraint)")
            return s, per_speed_maxp

    # no candidate passed: fallback to 1.0 (safe)
    print("No discrete speed met the target failure threshold; returning S_L = 1.0")
    return 1.0, per_speed_maxp

# --------------------------
# EDF+SRP Mode Simulator (integrates SRP & HI-switch-on-block)
# --------------------------
class EDFModeSimulatorSRP:
    def __init__(self, taskset, time_horizon=100.0, quantum=0.1, sl=0.5):
        self.tasks = taskset
        self.time_horizon = float(time_horizon)
        self.quantum = float(quantum)
        self.SL = float(sl)
        self.time = 0.0
        self.mode = 'LO'
        self.jobs = []
        self.ready = []
        self.next_release = {task: task.phase for task in self.tasks}
        self.log = []  # (t0, t1, job_or_None, mode)
        self.mode_switches = []
        self.total_switches = 0
        self.eps = 1e-9
        # SRP state
        self.locked = {}  # resource -> (holder_job, remaining_lock_time_in_wallclock)
        # compute resource ceilings (static)
        self.resource_ceiling = build_resource_ceilings(self.tasks)

    def cpu_speed(self):
        return 1.0 if self.mode == 'HI' else self.SL

    def current_system_ceiling(self):
        # returns -inf if no locked resources
        if not self.locked:
            return -math.inf
        # locked keys are resources; resource_ceiling[r] holds preemption-level ceilings
        return max(self.resource_ceiling.get(r, -math.inf) for r in self.locked.keys())

    def release_jobs_at(self, t):
        for task in self.tasks:
            while self.next_release[task] <= t + self.eps and self.next_release[task] < self.time_horizon + self.eps:
                if task.kind == 'LO':
                    sampled = task.sample_lo_execution()
                    job = Job(task, release_time=self.next_release[task], sampled_lo_work=sampled)
                else:
                    job = Job(task, release_time=self.next_release[task])
                self.jobs.append(job)
                self.ready.append(job)
                self.next_release[task] += task.period

    def drop_job(self, job, reason="dropped"):
        # finalize blocking time if currently blocked
        if job.block_start_time is not None:
            job.total_blocking_time += (self.time - job.block_start_time)
            job.block_start_time = None
        if job.dropped:
            return
        job.dropped = True
        job.finish_time = self.time
        if job in self.ready:
            self.ready.remove(job)

    def lock_resource_for_job(self, res, job, wallclock_lock_time):
        """Lock resource res for job for wallclock_lock_time seconds."""
        # store remaining lock time (wall-clock) for the holder
        self.locked[res] = (job, wallclock_lock_time)

    def release_resource(self, res):
        if res in self.locked:
            del self.locked[res]

    def switch_to_hi(self):
        if self.mode == 'HI':
            return
        t_s = self.time
        self.mode = 'HI'
        self.mode_switches.append((t_s, 'LO->HI'))
        # At t_s handle LO jobs per rules (as before)
        for job in list(self.ready):
            if job.task.kind != 'LO':
                continue
            Cdeg_effective = job.task.base_C_deg / max(self.SL, self.eps)
            if job.executed_time - Cdeg_effective > 1e-12:
                self.drop_job(job, reason='executed > C_deg at switch')
            else:
                if job.release_time < t_s - self.eps and job.deadline > t_s + self.eps:
                    allowed_remaining_time = max(0.0, Cdeg_effective - job.executed_time)
                    allowed_remaining_work = allowed_remaining_time * 1.0
                    allowed_total_work = job.work_done() + allowed_remaining_work
                    if allowed_total_work < job.base_work - self.eps:
                        job.remaining_work = min(job.remaining_work, allowed_remaining_work)
                    if job.remaining_work <= self.eps and (not job.is_complete()):
                        job.finish_time = t_s
                        if job in self.ready:
                            self.ready.remove(job)

    def maybe_switch_to_lo(self):
        pending = [j for j in self.ready if (not j.dropped) and (not j.is_complete()) and j.release_time <= self.time + self.eps]
        if len(pending) == 0 and self.mode == 'HI':
            self.mode = 'LO'
            self.mode_switches.append((self.time, 'HI->LO'))

    def preemption_level_job(self, job):
        # use 1/T of the job's task
        return 1.0 / job.task.period

    def pick_job_edf_srp(self):
        """Pick EDF among ready jobs, but enforce SRP: job's preemption level >= current_system_ceiling."""
        cand = [j for j in self.ready if (not j.dropped) and (not j.is_complete()) and j.release_time <= self.time + self.eps]
        if not cand:
            return None
        # Filter by SRP admission: preemption_level >= current system ceiling
        system_ceil = self.current_system_ceiling()
        admissible = [j for j in cand if self.preemption_level_job(j) >= system_ceil - 1e-12]
        if not admissible:
            # No admissible job due to SRP -> system idle (or deadlock).
            return None
        admissible.sort(key=lambda J: (J.deadline, J.release_time))
        return admissible[0]

    def run(self):
        t = 0.0
        while t < self.time_horizon - self.eps:
            self.time = t
            # decrement locks (if any) by how much wall-clock time has passed since last event?
            # We'll update lock timers as we advance time below when locks are held during non-preemptive CS.
            # release new jobs
            self.release_jobs_at(t)
            # drop jobs that missed deadlines before they execute
            for job in list(self.ready):
                if (not job.dropped) and (not job.is_complete()) and job.deadline <= t + self.eps:
                    self.drop_job(job, reason='deadline missed before execution')

            selected = self.pick_job_edf_srp()
            if selected is None:
                # nothing admissible now (either idle or SRP blocks all). Advance time by one quantum, but also
                # decrement any lock timers (shouldn't happen with our non-preemptive CS model unless a lock-holder is running).
                t_next = min(t + self.quantum, self.time_horizon)
                # log idle
                self.log.append((t, t_next, None, self.mode))
                # advance time and continue
                t = t_next
                # after idle, maybe switch back to LO
                self.maybe_switch_to_lo()
                continue

            # If the selected job wants to enter a CS (at job start) and next resource is locked by someone else,
            # then the job is blocked. If selected is HI and blocker is LO -> switch to HI immediately.
            next_cs = selected.next_pending_cs()
            if next_cs is not None:
                # resource is a string
                if next_cs in self.locked:
                    # Start blocking timer if not already started
                    if selected.block_start_time is None:
                        selected.block_start_time = t
                    holder_job, rem_lock = self.locked[next_cs]
                    # selected is blocked waiting for this resource
                    selected.blocked = True
                    # If HI job is blocked by LO job, switch to HI
                    if selected.task.kind == 'HI' and holder_job.task.kind == 'LO' and self.mode == 'LO':
                        # immediate mode switch
                        self.switch_to_hi()
                        # after switching to HI we may need to enforce drop/cap for LO jobs (already done inside switch_to_hi)
                        # Note: after switch to HI, SRP admission criteria uses system ceiling computed from locked resources.
                    # Cannot execute this quantum; log idle and advance time
                    self.log.append((t, t + self.quantum, None, self.mode))
                    t += self.quantum
                    # maybe switch back to LO at idle
                    self.maybe_switch_to_lo()
                    continue
                else:
                    # resource is free -> need to check admission again with SRP (we already did), so we can lock and execute CS non-preemptively
                    # But SRP requires that preemption level >= current system ceiling (we previously ensured)
                    # Acquire resource and run its CS non-preemptively to completion (or until time horizon)
                    cs_len_work = selected.cs_remaining.get(next_cs, 0.0)  # work units (at speed=1)
                    if cs_len_work <= 1e-12:
                        # weird case: nothing to do; mark CS as done
                        selected.cs_remaining[next_cs] = 0.0
                    else:
                        # If job was blocked earlier, accumulate blocking time
                        if selected.block_start_time is not None:
                            selected.total_blocking_time += (t - selected.block_start_time)
                            selected.block_start_time = None
                        # compute wallclock time to execute CS given current mode/speed
                        speed = self.cpu_speed()
                        wallclock_needed = cs_len_work / max(speed, self.eps)
                        # For non-preemptive CS we will allocate continuous wallclock_needed (but cap by horizon)
                        t_end = min(t + wallclock_needed, self.time_horizon)
                        dt = t_end - t
                        # Lock the resource for the job for 'dt' wall-clock time
                        self.lock_resource_for_job(next_cs, selected, dt)
                        # record start time if first execution
                        if selected.start_time is None:
                            selected.start_time = t
                            selected.total_waiting_time = t - selected.release_time
                        # perform the CS: consume cs_len_work (as work units)
                        selected.cs_remaining[next_cs] = max(0.0, selected.cs_remaining[next_cs] - (speed * dt))
                        # Also decrease remaining_work by same work amount (so total base_work accounted)
                        selected.remaining_work = max(0.0, selected.remaining_work - (speed * dt))
                        selected.executed_time += dt
                        # log
                        self.log.append((t, t_end, selected, self.mode))
                        # release the lock immediately because we've executed the CS fully (or until horizon)
                        # but if not finished (unlikely because we sized t_end to finish), keep remaining lock.
                        # For safety, check: if selected.cs_remaining[next_cs] <= eps -> release else decrement remaining lock
                        if selected.cs_remaining[next_cs] <= 1e-9:
                            # CS finished: release resource immediately
                            self.release_resource(next_cs)
                            # remove this resource from cs_order if done
                            # (we'll rely on next_pending_cs to skip zero CSs)
                        else:
                            # Partial CS (happens only if horizon cut): keep lock with remaining lock time t_end - t (but here we've used all dt)
                            # For consistency, store remaining lock time as 0 and leave locked entry; it will be cleaned at loop end.
                            # (Edge case; in practice horizon unlikely to cut a short CS.)
                            pass
                        # After CS completion, check if HI-trigger occurred: if this was a HI task executing and executed_time > C_thr (effective)
                        if selected.task.kind == 'HI' and self.mode == 'LO':
                            Cthr_eff = selected.task.base_C_thr / max(self.SL, self.eps)
                            if selected.executed_time - Cthr_eff > 1e-12:
                                # switch to HI at current time (t_end)
                                self.time = t_end
                                self.switch_to_hi()
                        # Move time forward
                        t = t_end
                        # After CS, if job completed (both CSs and remaining work), finalize
                        if selected.is_complete():
                            selected.finish_time = t
                            if selected in self.ready:
                                self.ready.remove(selected)
                        # Continue to next simulation step (do not consume extra quantum here)
                        continue

            # If no pending CS, execute ordinary (possibly preemptible) work for up to quantum or until completion
            # But SRP ensures selected.preemption_level >= current_system_ceiling
            speed = self.cpu_speed()
            t_end = min(t + self.quantum, self.time_horizon)
            dt = t_end - t
            work_done = speed * dt
            # If job will finish earlier, adjust dt to exact finishing time
            if selected.remaining_work <= work_done + self.eps:
                # need dt_needed wallclock time
                dt_needed = selected.remaining_work / max(speed, self.eps)
                t_end = t + dt_needed
                dt = dt_needed
                work_done = speed * dt
            if selected.start_time is None:
                selected.start_time = t
                selected.total_waiting_time = t - selected.release_time
            selected.remaining_work = max(0.0, selected.remaining_work - work_done)
            selected.executed_time += dt
            self.log.append((t, t_end, selected, self.mode))

            # After execution, check for HI trigger if this is a HI job in LO mode
            if selected.task.kind == 'HI' and self.mode == 'LO':
                Cthr_eff = selected.task.base_C_thr / max(self.SL, self.eps)
                if selected.executed_time - Cthr_eff > 1e-12:
                    # Switch to HI immediately at t_end
                    self.time = t_end
                    self.switch_to_hi()
                    # remove job from ready if done
                    if selected.is_complete():
                        selected.finish_time = t_end
                        if selected in self.ready:
                            self.ready.remove(selected)
                    # continue loop at same time (mode changed)
                    t = t_end
                    continue

            # If job finished its normal work, remove it
            if selected.is_complete():
                selected.finish_time = t_end
                if selected in self.ready:
                    self.ready.remove(selected)

            # After execution, drop jobs missing deadlines at or before t_end
            for job in list(self.ready):
                if (not job.dropped) and (not job.is_complete()) and job.deadline <= t_end + self.eps:
                    self.drop_job(job, reason='deadline missed after execution')

            # Advance time and loop
            t = t_end

        # Finalize simulation: drop incomplete jobs
        self.time = self.time_horizon
        for job in self.jobs:
            if (not job.dropped) and (not job.is_complete()):
                job.dropped = True
                job.finish_time = self.time

    # Reporting & plotting
    def print_summary(self):
        print("\n=== Simulation Summary (SRP) ===")
        print(f"Time horizon: {self.time_horizon}, QUANTUM: {self.quantum}, SL (LO speed): {self.SL}")
        print("Mode switches:")
        for t,desc in self.mode_switches:
            print(f"  t={t:.3f}: {desc}")
            self.total_switches += len(self.mode_switches)
        self.total_switches = len(self.mode_switches)
        print("\nJobs:")
        for job in sorted(self.jobs, key=lambda j: (j.task.name, j.release_time)):
            response_time = None
            if job.finish_time is not None:
                response_time = job.finish_time - job.release_time
            rt_str = f"{response_time:.3f}" if response_time is not None else "None"
            print(f"  {job} -> "
                f"RT={rt_str}, "
                f"Wait={job.total_waiting_time:.3f}, "
                f"Block={job.total_blocking_time:.3f}, "
                f"executed_time={job.executed_time:.3f}, ")
            status = "COMPLETED" if job.is_complete() and not job.dropped else ("DROPPED" if job.dropped else "INCOMPLETE")
            print(f"  {job} -> executed_time={job.executed_time:.3f}, work_done={job.work_done():.3f}, status={status}, start={job.start_time}, finish={job.finish_time}")
        misses = sum(1 for j in self.jobs if j.dropped)
        completes = sum(1 for j in self.jobs if (not j.dropped) and j.is_complete())
        print(f"\nTotal jobs: {len(self.jobs)}, completed: {completes}, dropped: {misses}")



    def get_summary(self):

        total_jobs = len(self.jobs)

        misses = sum(1 for j in self.jobs if j.dropped)

        completes = sum(1 for j in self.jobs if (not j.dropped) and j.is_complete())

        # ✅ Deadline Miss Ratio
        deadline_miss_ratio = misses / total_jobs if total_jobs > 0 else 0

        # ✅ Response Times (only completed jobs)
        response_times = [
            j.finish_time - j.release_time
            for j in self.jobs
            if j.finish_time is not None and not j.dropped
        ]

        avg_response_time = (
            sum(response_times) / len(response_times)
            if len(response_times) > 0 else 0
        )

        return {
            "total_jobs": total_jobs,
            "completed_jobs": completes,
            "missed_jobs": misses,
            "deadline_miss_ratio": deadline_miss_ratio,
            "avg_response_time": avg_response_time
        }

# --------------------------
# Helper to convert Task objects to analytic dicts
# --------------------------
def tasks_to_analytic_dicts(tasks):
    dicts = []
    for t in tasks:
        vals = [v for v,_ in t.pmf_lo]
        probs = [p for _,p in t.pmf_lo]
        d = {
            'T': float(t.period),
            'C_LO_vals': vals,
            'C_LO_p': probs
        }
        dicts.append(d)
    return dicts

# --------------------------
# Per-core S_L computation (ADDED)
# --------------------------
def compute_core_sl(core_tasks, core_max_speed):
    if not core_tasks:
        return core_max_speed, {}

    analytic = tasks_to_analytic_dicts(core_tasks)

    allowed_speeds = [s for s in SPEEDS if s <= core_max_speed]

    sl, info = find_S_L_analytic(
        analytic,
        speeds=allowed_speeds,
        F_s=F_S_target,
        res=RES
    )

    return sl, info

# --------------------------
# --------- ADDED: ENERGY POST-PROCESSING (no changes to scheduling code) ----------
# Energy constants you requested:
P_ind = 0.01    # Pi_n_d
C_ef  = 1.0     # C_ef
m_exp = 3       # m

def compute_energy_from_sim_log(sim, SL, P_ind=0.01, C_ef=1.0, m_exp=3):
    """
    Compute energy from sim.log entries.
    sim.log entries are tuples: (t0, t1, job_or_None, mode)
    - job_or_None: Job object or None (idle)
    - mode: 'LO' or 'HI' (scheduler mode during that segment)
    SL: the LO speed selected (used when mode == 'LO')
    Returns a dict with per-task energy, idle energy, energy_by_mode, and total energy.
    """
    per_task_energy = defaultdict(float)
    idle_energy = 0.0
    energy_by_mode = {'LO': 0.0, 'HI': 0.0}

    for seg in sim.log:
        t0, t1, job, mode = seg
        dt = max(0.0, t1 - t0)
        if dt <= 0:
            continue
        # determine speed used during this segment
        speed = 1.0 if mode == 'HI' else SL
        # instantaneous total power at this speed
        total_power = P_ind + C_ef * (speed ** m_exp)
        dE = total_power * dt
        # accumulate mode energy
        energy_by_mode[mode] = energy_by_mode.get(mode, 0.0) + dE
        if job is None:
            idle_energy += dE
        else:
            # accumulate per-task
            per_task_energy[job.task.name] += dE

    total_energy = sum(per_task_energy.values()) + idle_energy
    # sanity: also sum energy_by_mode
    total_by_mode = sum(energy_by_mode.values())
    return {
        'per_task_energy': dict(per_task_energy),
        'idle_energy': idle_energy,
        'energy_by_mode': energy_by_mode,
        'total_energy': total_energy,
        'total_by_mode': total_by_mode
    }

def print_energy_report_from_sim(sim, SL, P_ind=0.01, C_ef=1.0, m_exp=3):
    res = compute_energy_from_sim_log(sim, SL, P_ind=P_ind, C_ef=C_ef, m_exp=m_exp)
    print("\n================ ENERGY REPORT ================")
    print(f"Energy model params: P_ind={P_ind}, C_ef={C_ef}, m={m_exp}, SL={SL}")
    print("-----------------------------------------------")
    total = 0.0
    for task_name, e in sorted(res['per_task_energy'].items()):
        print(f" Task {task_name:>4}:  Energy = {e:.6f}")
        total += e
    print(f"\n Idle energy: {res['idle_energy']:.6f}")
    total += res['idle_energy']
    print("-----------------------------------------------")
    print(f" Energy while LO-mode: {res['energy_by_mode'].get('LO',0.0):.6f}")
    print(f" Energy while HI-mode: {res['energy_by_mode'].get('HI',0.0):.6f}")
    print("-----------------------------------------------")
    print(f" TOTAL ENERGY CONSUMED: {res['total_energy']:.6f}")
    # small sanity check
    if abs(res['total_energy'] - res['total_by_mode']) > 1e-9:
        print(f"Warning: total_by_mode ({res['total_by_mode']:.6f}) != total_energy ({res['total_energy']:.6f})")
    print("================================================\n")


def compute_total_blocking_time(all_sims):

    total_blocking = 0.0

    for sim in all_sims:
        for job in sim.jobs:

            # finalize unfinished blocking
            if job.block_start_time is not None:
                job.total_blocking_time += (sim.time - job.block_start_time)
                job.block_start_time = None

            total_blocking += job.total_blocking_time

    return total_blocking



def save_and_average_results_for_varying_tasks(
    num_tasks,
    deadline_miss_ratio,
    avg_response_time,
    total_energy,
    mode_switches,
    total_blocking_time,
    results_file="RESULTS.json",
    avg_results_file="RESULTS_AVG.json"
):

    # ================================
    # LOAD EXISTING RESULTS
    # ================================
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = []

    # Safety check
    if not isinstance(results, list):
        print("⚠️ Fixing invalid results format...")
        results = []

    # ================================
    # APPEND NEW ENTRY
    # ================================
    new_entry = {
        "num_tasks": num_tasks,
        "deadline_miss_ratio": deadline_miss_ratio,
        "avg_response_time": avg_response_time,
        "total_energy": total_energy,
        "mode_switches": mode_switches,
        "total_blocking_time": total_blocking_time
    }

    results.append(new_entry)

    # ================================
    # SAVE RAW RESULTS
    # ================================
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    # ================================
    # GROUP BY num_tasks
    # ================================
    grouped = defaultdict(list)

    for entry in results:
        grouped[entry["num_tasks"]].append(entry)

    # ================================
    # COMPUTE AVERAGES
    # ================================
    averaged_results = []

    for num_tasks, entries in grouped.items():

        avg_entry = {
            "num_tasks": num_tasks,
            "deadline_miss_ratio":
                sum(e["deadline_miss_ratio"] for e in entries) / len(entries),

            "avg_response_time":
                sum(e["avg_response_time"] for e in entries) / len(entries),

            "total_energy":
                sum(e["total_energy"] for e in entries) / len(entries),

            "mode_switches":
                sum(e["mode_switches"] for e in entries) / len(entries),

            "total_blocking_time":
                sum(e["total_blocking_time"] for e in entries) / len(entries)
        }

        averaged_results.append(avg_entry)

    # ================================
    # SORT RESULTS
    # ================================
    averaged_results.sort(key=lambda x: x["num_tasks"])

    # ================================
    # SAVE AVERAGED RESULTS
    # ================================
    with open(avg_results_file, "w") as f:
        json.dump(averaged_results, f, indent=4)

    print("✅ Results saved successfully.")



def save_and_average_results_for_varying_cores(
    num_cores,
    deadline_miss_ratio,
    avg_response_time,
    total_energy,
    mode_switches,
    total_blocking_time,
    results_file="RESULTS_RandomCores.json",
    avg_results_file="RESULTS_RandomCores_AVG.json"
):

    # ================================
    # LOAD EXISTING RESULTS
    # ================================
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = []

    # Safety check
    if not isinstance(results, list):
        print("⚠️ Fixing invalid results format...")
        results = []

    # ================================
    # APPEND NEW ENTRY
    # ================================
    new_entry = {
        "num_cores": num_cores,
        "deadline_miss_ratio": deadline_miss_ratio,
        "avg_response_time": avg_response_time,
        "total_energy": total_energy,
        "mode_switches": mode_switches,
        "total_blocking_time": total_blocking_time
    }

    results.append(new_entry)

    # ================================
    # SAVE RAW RESULTS
    # ================================
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    # ================================
    # GROUP BY num_tasks
    # ================================
    grouped = defaultdict(list)

    for entry in results:
        grouped[entry["num_cores"]].append(entry)

    # ================================
    # COMPUTE AVERAGES
    # ================================
    averaged_results = []

    for num_cores, entries in grouped.items():

        avg_entry = {
            "num_cores": num_cores,
            "deadline_miss_ratio":
                sum(e["deadline_miss_ratio"] for e in entries) / len(entries),

            "avg_response_time":
                sum(e["avg_response_time"] for e in entries) / len(entries),

            "total_energy":
                sum(e["total_energy"] for e in entries) / len(entries),

            "mode_switches":
                sum(e["mode_switches"] for e in entries) / len(entries),

            "total_blocking_time":
                sum(e["total_blocking_time"] for e in entries) / len(entries)
        }

        averaged_results.append(avg_entry)

    # ================================
    # SORT RESULTS
    # ================================
    averaged_results.sort(key=lambda x: x["num_cores"])

    # ================================
    # SAVE AVERAGED RESULTS
    # ================================
    with open(avg_results_file, "w") as f:
        json.dump(averaged_results, f, indent=4)

    print("✅ Results saved successfully.")



# --------------------------
# Main: generate tasks, find S_L, run EDF+SRP simulator
# --------------------------
if __name__ == "__main__":
    start_time = pytime.time()
    tasks = generate_random_taskset(n_tasks=NUM_TASKS, seed=RANDOM_SEED)
    # SAVE once
    #save_taskset_to_file(tasks, "taskset_25C_100T_RandomCores.json")

    #tasks = load_taskset_from_file("taskset_10C_100T.json")


    #analytic_tasks = tasks_to_analytic_dicts(tasks)

    # Diagnostic at s=1.0
    #t_list = build_time_check_list(analytic_tasks)
    #print("\nTime-check list (t):", t_list)
    #print("\nDiagnostic: computing Pr(DBF(t)>t) at s = 1.0 ...")
    #probs_at_1 = compute_prob_exceed_for_t(analytic_tasks, 1.0, t_list, res=RES)
    #if probs_at_1:
     #   for t,p in sorted(probs_at_1.items()):
      #      print(f" t={t:.3f} -> Pr(DBF(t)>t) = {p:.6e}")
       # print("max at s=1.0:", max(probs_at_1.values()))
    #else:
     #   print("No t values to check.")

    # Find S_L analytically
    #SL_found, per_speed_info = find_S_L_analytic(analytic_tasks, speeds=SPEEDS, F_s=F_S_target, res=RES)
    #print(f"\nUsing S_L = {SL_found} for simulation.")

    # Build resource ceilings table and print
    #resource_ceilings = build_resource_ceilings(tasks)
    #print("\nResource ceilings (preemption-level based):")
    #for r, c in resource_ceilings.items():
     #   print(f" {r}: {c:.6f}")

    # Run EDF + SRP simulation
    #sim = EDFModeSimulatorSRP(taskset=tasks, time_horizon=TIME_HORIZON, quantum=QUANTUM, sl=SL_found)
    #sim.run()
    #sim.print_summary()
    #sim.plot_gantt()

    # -------------------- ADDED: print energy report computed from sim.log --------------------
    #print_energy_report_from_sim(sim, SL_found, P_ind=P_ind, C_ef=C_ef, m_exp=m_exp)

# --------------------------
# MULTICORE EXECUTION
# --------------------------

core_tasksets = partition_tasks_to_cores(tasks, NUM_CORES)

total_energy = 0.0
all_sims = []

# results = {
#     "cores": [],
#     "total_energy": 0,
#     "execution_time": 0,
#     "mode_switches": 0,
#     "total_blocking_time": 0
# }


all_sims = []
total_energy = 0
total_switches = 0.0
total_blocking_time = 0

for cid in range(NUM_CORES):

    core_tasks = core_tasksets[cid]

    print("\n================================")
    print(f" CORE {cid}")
    print(" Tasks:", [t.name for t in core_tasks])

    sl_core, _ = compute_core_sl(core_tasks, 1.0)
    print(" Selected S_L:", sl_core)

    sim = EDFModeSimulatorSRP(
        taskset=core_tasks,
        time_horizon=TIME_HORIZON,
        quantum=QUANTUM,
        sl=sl_core
    )

    sim.run()
    sim.print_summary()

    energy = compute_energy_from_sim_log(
        sim,
        sl_core,
        P_ind=P_ind,
        C_ef=C_ef,
        m_exp=m_exp
    )

    print(f" Core {cid} energy = {energy['total_energy']:.6f}")

    total_energy += energy['total_energy']
    total_switches += sim.total_switches
    print(" total_switches :", total_switches)


    # ✅ ONLY store simulator (NOT results yet)
    all_sims.append(sim)

total_blocking_time += compute_total_blocking_time(all_sims)
print("\n================================")
print(f" TOTAL BLOCKING TIME = {total_blocking_time:.6f}")
print("================================")

print("\n================================")
print(f" TOTAL SYSTEM ENERGY = {total_energy:.6f}")
print("================================")
print(f"\nElapsed wall time: {pytime.time() - start_time:.2f} s")


# results["total_energy"] = total_energy
# results["execution_time"] = pytime.time() - start_time
# results["mode_switches"] = total_switches
# results["total_blocking_time"] = total_blocking_time



# 🔷 Collect all jobs from all cores
all_jobs = []
for sim in all_sims:
    all_jobs.extend(sim.jobs)

total_jobs = len(all_jobs)

misses = sum(1 for j in all_jobs if j.dropped)

response_times = [
    j.finish_time - j.release_time
    for j in all_jobs
    if j.finish_time is not None and not j.dropped
]

deadline_miss_ratio = misses / total_jobs if total_jobs > 0 else 0

avg_response_time = (
    sum(response_times) / len(response_times)
    if len(response_times) > 0 else 0
)

# file_name = "RESULTS.json"

# # Load existing results
# if os.path.exists(file_name):
#     with open(file_name, "r") as f:
#         results = json.load(f)
# else:
#     results = []


# # 🔴 safety check (prevents your current error forever)
# if not isinstance(results, list):
#     print("⚠️ Fixing results format...")
#     results = []

# # Append new run
# results.append({
#     "num_tasks": NUM_TASKS,
#     "deadline_miss_ratio": deadline_miss_ratio,
#     "avg_response_time": avg_response_time,
#     "total_energy": total_energy,
#     "mode_switches": total_switches,
#     "total_blocking_time": total_blocking_time
# })

# # Save back
# with open(file_name, "w") as f:
#     json.dump(results, f, indent=4)


# # Group entries by num_tasks
# grouped = defaultdict(list)

# file_name_avg = "RESULTS_AVG.json"

# for entry in results:
#     grouped[entry["num_tasks"]].append(entry)

# # Compute averages
# # Load existing results
# # if os.path.exists(file_name_avg):
# #     with open(file_name_avg, "r") as f:
# #         averaged_results = json.load(f)
# # else:
# averaged_results = []

# for num_tasks, entries in grouped.items():

#     avg_entry = {
#         "num_tasks": num_tasks,
#         "deadline_miss_ratio": sum(e["deadline_miss_ratio"] for e in entries) / len(entries),
#         "avg_response_time": sum(e["avg_response_time"] for e in entries) / len(entries),
#         "total_energy": sum(e["total_energy"] for e in entries) / len(entries),
#         "mode_switches": sum(e["mode_switches"] for e in entries) / len(entries),
#         "total_blocking_time": sum(e["total_blocking_time"] for e in entries) / len(entries)
#     }

#     averaged_results.append(avg_entry)

# # Sort by num_tasks
# averaged_results.sort(key=lambda x: x["num_tasks"])

# # Write averaged data to new JSON file
# with open(file_name_avg, "w") as f:
#     json.dump(averaged_results, f, indent=4)


# save_and_average_results_for_varying_tasks(
#     num_tasks=NUM_TASKS,
#     deadline_miss_ratio=deadline_miss_ratio,
#     avg_response_time=avg_response_time,
#     total_energy=total_energy,
#     mode_switches=total_switches,
#     total_blocking_time=total_blocking_time
# )


save_and_average_results_for_varying_cores(
    num_cores=NUM_CORES,
    deadline_miss_ratio=deadline_miss_ratio,
    avg_response_time=avg_response_time,
    total_energy=total_energy,
    mode_switches=total_switches,
    total_blocking_time=total_blocking_time
)