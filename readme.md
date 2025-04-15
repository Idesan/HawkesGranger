# Cardinality-Regularized Hawkes-Granger Model

**Tsuyoshi (Ide-san) Ide**

IBM Thomas J. Research Center, tide@us.ibm.com / ide@ide-research.net

## What is this repository about?

This repository provides an end-to-end demo of a Granger-causal analysis approach proposed in

> Tsuyoshi Idé, Georgios Kollias, Dzung T. Phan, Naoki Abe, "[Cardinality-Regularized Hawkes-Granger Model](https://proceedings.neurips.cc/paper/2021/hash/15cf76466b97264765356fcc56d801d1-Abstract.html)," Advances in Neural Information Processing Systems 34 (NeurIPS 2021). ([slides](https://neurips.cc/media/neurips-2021/Slides/27855_O3GiVn6.pdf)).

[This Jupyter notebook](demo_L0Hawkes.ipynb) provides full details (tested on 06/10/2024 on a clean pip environment with libraries described in `requirements.txt`). Enjoy!


## Dependency

The core module `L0HawkesLib.py` depends **only on `numpy`**. This is the only critical dependency of this Python implementation of the *L0Hawkes* algorithm.

For visualization purposes, a utility library called `L0HawkesVisualize.py` has been included, which depends on:
- `pandas`
- `matplotlib`
- `seaborn`

If you want to reproduce the notebook as it is, I suggest running
```python
pip install numpy pandas matplotlib seaborn
```
on a clean pip environment so all the sub-libraries will be automatically installed.

If you are not comfortable with installing these, just download only `L0HawkesLib.py` and use it in your environment. 
