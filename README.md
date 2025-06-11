# Scheduling of Distributed LLM Serving on Heterogeneous GPUs

Python code to accompany the thesis report.

## Core Code Contents:

1. **`main_divider.py`**: Main scheduler using the divider island approach (executable - contains several experiments).
2. **`main_direct.py`**: Main scheduler using the direct island approach (executable - contains several experiments).
3. **`packer.py`**: Inner loop for packing tasks onto GPU islands (executable - contains test script).
4. **`evaluator.py`**: Evaluator for measuring performance metrics.
5. **`simulator.py`**: Simulator. Based off [Shallowsim](https://github.com/icezack12/shallowsim).
6. **`divider.py`**: Island divider logic.

## Additional Files:

1. **`traces/`**: Contains workload traces used for evaluation (Based off [Azure LLM](https://github.com/Azure/AzurePublicDataset))
2. **`data/`**: Contains the results of the evaluation scripts.
3. **`notebooks/`**: Contains Jupyter notebooks for analysis and visualization of results.