# QAOA Max-Cut

Implementing and testing QAOA for the [Max-Cut problem](https://quantum.cloud.ibm.com/learning/en/courses/variational-algorithm-design/examples-and-applications#optimization-max-cut)

## Usage

This program can run a test for Max-cut in various platforms: a local Qiskit emulator, [Quantum Inspire](https://www.quantum-inspire.com), and [IBM Cloud](https://quantum.cloud.ibm.com).

In `main.py` you can modify various parameters, to run different tests:
- `N` and `EDGES`: defines the graph that Max-Cut is being run on, as an adjacency list (see example already graph in `main.py`)
- `REPS` is the number of repetitions of the optimizer circuit
- `MAX_ITER` is the number of iterations of the sampler
- `OPTIMIZER_NUM_SHOTS` is the number of times each circuit should be executed for each run of the sampler
- `TOL` is the tolerance of the optimizer
- `NODE_GROUPING_NUM_SHOTS` is the number of times the circuit should be executed, for the estimator run
- `TEST_NAME` is the name of the folder where the results should end up (this should be different for each run)

Then you can run the testing on various platforms
- To run it locally, set `LOCAL=True`, and leave the rest of the con
- To run on cloud platforms, set `LOCAL=False`
    - To run it on QI, set `PLATFORM="QI"` and set `BACKEND_NAME` and `QUBIT_PRIORITY` as you wish (see examples in `main.py`)
    - To run it on IBM, set `PLATFORM="IBM"`, get a token and CRN for `MY_TOKEN` and `MY_CRN`
        - To run in the Fez simulator set `IBM_SIM=True`
        - To run in the least busy real processor, set `IBM_SIM=False`

In each run, a folder with the name of the test in `./test_results` is created.

This new folder contains:
- `config_{TEST_NAME}.txt` characteristics of the test
- `initial_graph_{TEST_NAME}.jpg` the initial graph
- `histogram_{TEST_NAME}.jpg` the histogram with the measurements in the final circuit
- `coloured_graph_{TEST_NAME}.jpg` final division of the nodes  
- `convergence_{TEST_NAME}.jpg` the graphic with the cost evaluated in each iteration

## Setup

### Install uv

Ideally `pipx`
```bash
pipx install uv
```

`pip` also works
```
pip install uv
```

Or [see docs for all options](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

### Install dependencies
```bash
uv sync
```
