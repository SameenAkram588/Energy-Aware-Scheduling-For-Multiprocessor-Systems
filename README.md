# Energy-Aware-Scheduling-For-Multiprocessor-Systems

# Code Walkthrough

This document explains how the simulator executes, what each major method does, and how different components interact.

# Program Execution Flow

When the program starts, Python executes the statements at the bottom of the file.

The main execution begins by calling:

```python
save_and_average_results_for_varying_cores(...)
```

(or `save_and_average_results_for_varying_tasks(...)`, depending on the experiment being performed.)

These functions act as the **entry point** of the simulation.

---

# Step 1 – Generate / Load Task Set

Before running the scheduler, the program prepares the workload.

Depending on the experiment, one of the following methods is used:

```python
generate_random_taskset()
```

Creates a random set of tasks using the configured parameters.

or

```python
load_taskset_from_file()
```

Loads a previously saved task set from a JSON file.

Each generated task is stored as a **Task** object.

---

# Step 2 – Create Task Objects

Each task is represented by the `Task` class.

The constructor initializes information such as:

* Task ID
* Period
* Deadline
* Criticality level
* Worst-case execution times
* Resources required
* Critical sections
* Phase

The `Task` object only describes the periodic task.

It does **not** represent an executing instance.

---

# Step 3 – Create Job Objects

Whenever a task reaches its release time, the scheduler creates a new

```python
Job(...)
```

object.

A Job represents one execution instance of a periodic task.

Each Job stores

* Release time
* Remaining execution time
* Deadline
* Current execution state
* Blocking information
* Completion status

Multiple Job objects can exist for the same Task.

---

# Step 4 – Build Resource Ceilings

Before scheduling begins,

```python
build_resource_ceilings()
```

is called.

This function computes the ceiling priority for every shared resource.

The calculated ceilings are later used by the SRP protocol to determine whether a task can lock a resource.

---

# Step 5 – Partition Tasks

For multicore execution,

```python
partition_tasks_to_cores()
```

assigns tasks among the available processor cores.

Each core receives its own task list.

A separate scheduler is then created for every core.

---

# Step 6 – Create Scheduler

Each processor core creates an instance of

```python
EDFModeSimulatorSRP
```

This is the main scheduling engine.

The constructor initializes

* Simulation clock
* Current mode (LO or HI)
* Ready queue
* Released jobs
* Release times
* Execution log
* Mode switch counter
* Scheduler parameters

After initialization, the scheduler waits for the `run()` method.

---

# Step 7 – Run Simulation

The simulation begins by calling

```python
run()
```

inside the scheduler.

This method repeatedly performs the following operations until the simulation time reaches the specified time horizon.

---

## Release New Jobs

The scheduler checks whether any task should release a new job.

If yes,

* a Job object is created
* the job is inserted into the ready queue

---

## Select Next Job

The scheduler calls

```python
pick_job_edf()
```

This function selects the job having the earliest deadline.

Before selecting, it also verifies SRP blocking conditions.

If a resource ceiling prevents execution, the job remains blocked.

---

## Execute Job

The selected job executes for one scheduling quantum.

During execution the scheduler

* decreases remaining execution time
* updates current simulation time
* records execution in the log
* checks critical sections
* updates resource ownership

---

## Check Completion

If the remaining execution time reaches zero,

the job is marked as completed and removed from the ready queue.

Completion statistics are updated.

---

## Detect Deadline Miss

The scheduler checks whether any active job has exceeded its deadline.

Deadline misses are recorded for later performance analysis.

---

## Handle Mode Switching

If execution exceeds the LO execution budget,

the scheduler switches from

LO Mode

to

HI Mode.

Mode switches are counted using

```python
maybe_switch_to_hi()
```

Similarly,

```python
maybe_switch_to_lo()
```

returns the scheduler to LO mode when appropriate.

---

# Step 8 – Simulation Ends

When the simulation reaches the configured time horizon,

the scheduler exits the `run()` loop.

Performance statistics are then computed.

---

# Step 9 – Compute Metrics

Several helper functions calculate the final performance.

These include

* Deadline Miss Ratio
* Average Response Time
* Energy Consumption
* Total Blocking Time
* Number of Mode Switches

Energy is calculated using

```python
compute_energy_from_sim()
```

Blocking time is calculated using

```python
compute_total_blocking_time()
```

---

# Step 10 – Store Results

The experiment results are passed to

```python
save_and_average_results_for_varying_cores()
```

or

```python
save_and_average_results_for_varying_tasks()
```

These functions

* save simulation data
* compute averages across multiple runs
* write the results into JSON files

These JSON files are later used for plotting.

---

# Graph.py Execution

`Graph.py` is independent from the scheduler.

Execution flow:

1. Load JSON result files.
2. Extract performance metrics.
3. Prepare x-axis and y-axis data.
4. Smooth curves using interpolation (where required).
5. Plot graphs with Matplotlib.
6. Save figures for thesis and presentation.

No scheduling is performed in this file.

---

# Overall Call Sequence

```text
Program Starts
      │
      ▼
Generate / Load Task Set
      │
      ▼
Create Task Objects
      │
      ▼
Build Resource Ceilings
      │
      ▼
Partition Tasks to CPU Cores
      │
      ▼
Create EDFModeSimulatorSRP
      │
      ▼
run()
      │
      ├── Release Jobs
      ├── Create Job Objects
      ├── Pick EDF Job
      ├── Apply SRP
      ├── Execute Quantum
      ├── Update Ready Queue
      ├── Mode Switching
      ├── Deadline Checking
      └── Completion Checking
      │
      ▼
Compute Metrics
      │
      ▼
Save JSON Results
      │
      ▼
Graph.py Reads JSON
      │
      ▼
Generate Performance Graphs
```
