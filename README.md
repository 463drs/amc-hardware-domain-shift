# AMC under receiver hardware impairments

Investigating how hardware impairments of a low-cost SDR receiver 
(RTL-SDR, R828D tuner) degrade deep-learning modulation classifiers 
trained on synthetic data (RadioML 2018.01A), and whether calibrating 
the training-data impairment model to measured receiver characteristics 
closes the sim-to-real gap.

## Setup
...
## Reproduce
python src/train.py --config configs/baseline.yaml