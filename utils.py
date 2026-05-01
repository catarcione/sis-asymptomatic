from collections import defaultdict

def sum_first_t(dictionary, t):
    """
    Aggregate node scores across time steps for temporal epidemic analysis.
    
    Computes cumulative scores for each node by summing metric values from the first
    t+1 time steps (0 through t inclusive). This temporal aggregation identifies nodes
    that are consistently central or important throughout the epidemic evolution, rather
    than just at a single snapshot.
    
    Parameters
    ----------
    dictionary : dict of dict
        Nested dictionary structure containing time-indexed node scores:
        - Outer keys: time step indices (integers 0, 1, 2, ...)
        - Inner keys: node IDs (any hashable type)
        - Values: metric scores (numeric, typically float)
        Format: {0: {node1: score, node2: score, ...},
                 1: {node1: score, node2: score, ...},
                 ...}
    t : int
        Last time step to include in aggregation (inclusive). Must be a valid key
        in the outer dictionary (i.e., 0 ≤ t < total_time_steps). The function sums
        scores from time steps 0, 1, 2, ..., t, giving a total of t+1 time steps.
    
    Returns
    -------
    dict
        Dictionary mapping each node ID to its cumulative score (float) across the
        specified time window. Only includes nodes that appear in at least one of
        the time steps 0 through t:
        - Nodes present in all time steps: sum of all their scores;
        - Nodes missing from some steps: sum of only the steps where they appear;
        - Nodes not appearing in any step: not included in result.
    """
    
    result = defaultdict(float)

    # Iterate through time steps from 0 to t (inclusive)
    for i in range(t + 1):
        # Add each node's score at time step i to the cumulative total
        for key, score in dictionary[i].items():
            result[key] += score

    # Convert defaultdict back to regular dict for cleaner output
    return dict(result)


def compute_performance_metrics(predicted_set, true_set):
    """
    Calculate precision, recall, and F1 score for set-based prediction.
    
    Evaluates how well a predicted set matches a ground truth set using standard
    binary classification metrics. Used to assess epidemic source identification
    algorithms that predict which nodes are infected/asymptomatic.
    
    Parameters
    ----------
    predicted_set : set
        Set of node IDs predicted as positive (e.g., predicted asymptomatic infections).
    true_set : set
        Ground truth set of node IDs that are actually positive (e.g., true
        asymptomatic infections).

    Returns
    -------
    dict
        Dictionary containing performance metrics:
        'precision' : float
            Proportion of predictions that are correct. Range [0, 1].
            Formula: TP / (TP + FP)
            Returns 0 if no predictions made (TP + FP = 0).
        'recall' : float
            Proportion of true positives that were predicted. Range [0, 1].
            Formula: TP / (TP + FN)
            Returns 0 if no true positives exist (TP + FN = 0).
        'f1' : float
            Harmonic mean of precision and recall. Range [0, 1].
            Formula: 2 × precision × recall / (precision + recall)
            Returns 0 if both precision and recall are 0.
    """
    
    tp = len(predicted_set & true_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)
    
    # Compute precision: what fraction of predictions are correct?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Compute recall: what fraction of true positives were found?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Compute F1: harmonic mean of precision and recall
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_centrality_metrics(network, snapshots, asymptomatic):
    """
    Compute centrality metrics for all snapshots in an epidemic simulation.
    
    For each snapshot, calculates observed betweenness centrality and contact scores,
    maintaining cumulative observations across time steps.
    
    Parameters
    ----------
    network : NetworkX graph
        The contact network on which the epidemic spreads.
    snapshots : list of set
        List of infected node sets at different time points.
    asymptomatic : set
        Set of nodes that are asymptomatic (unobservable when infected).
    
    Returns
    -------
    tuple of (dict, dict, dict)
        - observed_betweenness: dict mapping time step to betweenness centrality scores
        - contacts: dict mapping time step to contact scores
        - observations: dict mapping time step to cumulative observed infected nodes
    """
    from centrality import observed_betweenness, contact
    
    observed_betws = {}
    contacts = {}
    observations = {}
 
    for t, I_t in enumerate(snapshots):
        # Observed infected = total infected - asymptomatic
        O_t = I_t - asymptomatic
        
        # Compute observed betweenness for current snapshot
        obs_betw = observed_betweenness(network, O_t)
        observed_betws[t] = obs_betw
 
        # Accumulate observations across time
        if t > 0:
            O_t = O_t | observations[t-1]
        observations[t] = O_t
 
        # Compute contact scores
        cont = contact(network, O_t)
        contacts[t] = cont
    
    return observed_betws, contacts, observations
 
 
def evaluate_prediction_methods(network, snapshots, asymptomatic, 
                                 observed_betweenness, contacts, observations,
                                 fractions):
    """
    Evaluate multiple prediction methods for identifying asymptomatic infections.
    
    Compares three approaches:
    1. Naive baseline: predict all unobserved nodes as asymptomatic
    2. Observed betweenness: rank by cumulative betweenness centrality
    3. Contact score: rank by fraction of observed infected neighbors
    
    Parameters
    ----------
    network : NetworkX graph
        The contact network on which the epidemic spreads.
    snapshots : list of set
        List of infected node sets at different time points.
    asymptomatic : set
        Ground truth set of asymptomatic nodes.
    observed_betweenness : dict
        Time-indexed dictionary of betweenness centrality scores.
    contacts : dict
        Time-indexed dictionary of contact scores.
    observations : dict
        Time-indexed dictionary of cumulative observed nodes.
    fractions : list of float
        Fractions of asymptomatic nodes to predict (e.g., [0.1, 0.5, 1.0]).
    
    Returns
    -------
    tuple of (dict, dict, dict)
        - naive_scores: performance metrics for naive baseline at each time step
        - betweenness_scores: performance metrics for betweenness method at different fractions
        - contact_scores: performance metrics for contact method at different fractions
    """
    
    num_asymp = len(asymptomatic)
    top_ks = [int(frac * num_asymp) for frac in fractions]
    
    naive_scores = {}
    obs_betw_scores = {}
    contact_scores = {}
 
    for t, _ in enumerate(snapshots):
        obs = observations[t]
 
        # Get cumulative betweenness and current contact scores
        betw = sum_first_t(observed_betweenness, t)
        conts = contacts[t]
 
        # Get candidate nodes (unobserved nodes)
        candidates = [node for node in network.nodes() if node not in obs]
 
        # Extract scores for candidates only
        betw_eval = {i: betw[i] for i in candidates}
        cont_eval = {i: conts[i] for i in candidates}
 
        # Rank candidates by each metric
        betw_rank = dict(sorted(betw_eval.items(), key=lambda x: x[1], reverse=True))
        cont_rank = dict(sorted(cont_eval.items(), key=lambda x: x[1], reverse=True))
 
        # Evaluate at different prediction sizes
        betw_results = {}
        cont_results = {}
 
        for frac, k in zip(fractions, top_ks):
            # Get top-k predictions
            betw_top = set(list(betw_rank.keys())[:k])
            cont_top = set(list(cont_rank.keys())[:k])
 
            # Compute performance metrics
            betw_results[f"top-{frac}"] = compute_performance_metrics(betw_top, asymptomatic)
            cont_results[f"top-{frac}"] = compute_performance_metrics(cont_top, asymptomatic)
 
        # Naive baseline: predict all unobserved nodes as asymptomatic
        naive_pred = set(network.nodes()) - set(obs)
        naive_metrics = compute_performance_metrics(naive_pred, asymptomatic)
 
        # Store results
        naive_scores[f"{t+1} snapshots"] = naive_metrics
        obs_betw_scores[f"{t+1} snapshots"] = betw_results
        contact_scores[f"{t+1} snapshots"] = cont_results
    
    return naive_scores, obs_betw_scores, contact_scores