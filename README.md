# sis-asymptomatic
This repository provides the code to perform the simulations described in the paper:

**"Identifying Asymptomatic Nodes in SIS Network Epidemics Using Betweenness Centrality over Time"**

*Conrado Catarcione Pinto, Vitor Martins Gouvêa, Daniel Ratton Figueiredo*

# Overview
Asymptomatic individuals play a significant role in the spread of epidemics, yet they often remain undetected. This work implements and evaluates methods for **identifying asymptomatic nodes** in SIS network-based epidemic simulations, using only the network topology and snapshots of observed symptomatic cases.

The aim of this work is to evaluate the Observed Betweenness metric proposed in [DOI](https://doi.org/10.5753/wperformance.2024.2414) within the context of the SIS epidemic model.

# Modules
## `epidemic.py`
This module is responsible for **epidemic simulation** and the generation of **temporal snapshots** in an SIS model.

- **`choose_asymptomatic_nodes`** randomly selects nodes to be asymptomatic (unobservable when infected) with a specified fraction.

- **`epidemic_step`** executes a single time step of the SIS epidemic model, where infected nodes attempt to infect susceptible neighbors with probability β (beta) and recover with probability γ (gamma).

- **`burn_in`** runs the epidemic for a burn-in period to reach a quasi-steady state, removing transient effects from initial conditions.

- **`collect_snapshots`** records epidemic state at regular time intervals, capturing multiple observations from the endemic state for analysis.

This module provides the **ground-truth infected sets** at different time points.

## `centrality.py`
This module implements different **node-ranking strategies** based on network topology and observed infection data.

- **`observed_betweenness`** computes a modified betweenness centrality, considering only shortest paths between observed infected nodes. This identifies nodes that act as bridges within the observed infection subnetwork.

- **`contact`** computes, for each node, the fraction of its neighbors that are in the observed infected set. This captures local exposure to observed infections and identifies nodes with high contact to the epidemic.

- **`naive_baseline`** implements a trivial prediction strategy where all unobserved nodes are predicted as infected, providing a baseline for evaluating more sophisticated methods.

These measures assign a score to each node, interpreted as a likelihood of being asymptomatic. The resulting scores are used for ranking and evaluation.

## `utils.py`
This module provides **utility functions** for temporal aggregation and performance evaluation.

- **`sum_first_t`** aggregates node scores across time steps by summing metric values from the first t snapshots. This temporal aggregation identifies nodes that are consistently central throughout the epidemic evolution.

- **`compute_performance_metrics`** calculates precision, recall, and F1 score for set-based predictions, evaluating how well predicted asymptomatic sets match ground truth.

- **`compute_centrality_metrics`** computes centrality metrics (observed betweenness and contact scores) for all snapshots in an epidemic simulation, maintaining cumulative observations across time steps.

- **`evaluate_prediction_methods`** compares three prediction approaches:
  - **Naive baseline**: predict all unobserved nodes as asymptomatic
  - **Observed betweenness**: rank by cumulative betweenness centrality
  - **Contact score**: rank by fraction of observed infected neighbors
  
  It returns performance metrics (precision, recall, F1) at different prediction sizes.

This module enables quantitative comparison of different strategies and temporal analysis of epidemic data.

See `example.ipynb` for a complete workflow including parameter configuration, simulation, and evaluation with detailed results display.

# Contact
Questions or feedback? Please contact:
conrado@cos.ufrj.br