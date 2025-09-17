# A Time-Reversal Control Synthesis for Steering the State of Stochastic Systems

This repository is created by Yuhang Mei and contains the Python source code to reproduce the experiments in our paper [A Time-Reversal Control Synthesis for Steering the State of Stochastic Systems] ([https://arxiv.org/pdf/2410.04615](https://arxiv.org/abs/2504.00238)).


## Setup
* Python/ Numpy, Scipy, Matplotlib
* Pytorch

## Running the code and regenerating data and figures.
1. For Two-dimensional Brownian Bridge example, run the 'brownian_bridge.py' to generate and save the data. We already update the data and the figures for the results in paper. You can play with different time step size, sample size. Use 'plot_result.ipynb' to load the data and plot figures.
2. For Inverted Pendulum and Extended Inverted Pendulum example, run the 'inverted_pendulum.py' and 'inverted_pendulum_3d.py' to generate and save the data. We already update the data and the figures for the results in paper. Use 'plot_result.ipynb' to load the data and plot figures.