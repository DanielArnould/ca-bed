"""
This module contains all necessary tools for evaluating a run.
It should **ONLY** operate on saved records of runs and it
should output the same metrics used by the original UoT paper:
- Success rate
- Mean Conversation Length
- Mean Conversation Length in Succesful Cases

It should also create some helpful diagnostics such as:
- Mean Depth
- Max Depth
- Node count

A reasonable first approach would be to first design a schema for how outputs and the tree should be saved
so they can be parsed here again.
"""
