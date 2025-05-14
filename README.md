# Unifying Problems

This code reproduces all the experiments of the paper "On the interconnections between Calibration, Quantification, and Classifier Accuracy Prediction under Dataset Shift" currently under submission.


## Installation

```bash
conda create -n unifyingenv python=3.11
conda activate unifyingenv
git clone git@github.com:AlexMoreo/UnifyingProblems.git
cd UnifyingProblems
pip install -r requirements.txt
```

## Scripts

The scripts that reproduce all the experiments are listed in the root folder as:
_[taskname]\_experiments\_[type-of-shift].py_; for example [calibration_experiments_label_shift.py](calibration_experiments_label_shift.py)



