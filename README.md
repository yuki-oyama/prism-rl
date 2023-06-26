# Prism-constrained Recursive Logit Model
Python code for the estimation of a prism-constrained recursive logit (Prism-RL) model.

## Paper
For more details, please see the paper

Oyama, Y. (2023) [Capturing positive network attributes during the estimation of recursive logit models: A prism-based approach](https://www.sciencedirect.com/science/article/pii/S0968090X23000037?via%3Dihub). Transportation Research Part C: Emerging Technologies 147, 104014.

If you find this code useful, please cite the paper:
```
@article{oyama2023prism,
  title={Capturing positive network attributes during the estimation of recursive logit models: A prism-based approach},
  author={Oyama, Yuki},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={147},
  pages={104014},
  year={2023},
  publisher={Elsevier}
}
```

## Quick Start
Estimate a Prism-RL model using synthetic observations in the Sioux Falls network.

```
python run_estimation.py --rl True --prism True --n_samples 1 --test_ratio 0
```

For validation, you need to split the data in more than one samples and set test ratio greater than zero.

```
python run_estimation.py --rl True --prism True --n_samples 10 --test_ratio 0.2
```
