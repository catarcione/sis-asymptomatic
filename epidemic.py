import random

def choose_asymptomatic_nodes(graph, fraction, seed=None):
    """
    Randomly select nodes to be asymptomatic in an epidemic simulation.
        
    Parameters
    ----------
    graph : NetworkX graph
        The contact network on which the epidemic spreads. Nodes represent individuals
        and edges represent contact relationships through which infection can spread.
    fraction : float
        Proportion of nodes to designate as asymptomatic, in range [0, 1].
    seed : int or None, optional
        Random seed for reproducible selection. If None (default), selection is
        non-deterministic and will vary between calls. Use a fixed integer for
        reproducible experiments and testing.
    
    Returns
    -------
    set
        Set of node IDs designated as asymptomatic. The size of this set
        is floor(fraction × number_of_nodes). Nodes are selected uniformly at random
        without replacement.
    """
    
    random.seed(seed)
    nodes = list(graph.nodes())
    k = int(len(nodes) * fraction)  # Calculate the number of asymptomatic nodes

    return set(random.sample(nodes, k))


def epidemic_step(graph, beta, gamma, infected):
    """
    Execute a single time step of the SIS (Susceptible-Infected-Susceptible) epidemic model.
    In this model, recovered individuals do not gain immunity
    and immediately return to the susceptible state.
    
    In each step:
    1. Infected nodes attempt to infect their susceptible neighbors with probability beta
    2. Infected nodes recover (become susceptible again) with probability gamma
    
    Parameters
    ----------
    graph : NetworkX graph
        The contact network on which the epidemic spreads. Nodes represent individuals
        and edges represent contact relationships through which infection can spread.
    beta : float
        Transmission probability per contact, in range [0, 1]. At each iteration, an
        infected node attempts to infect each susceptible neighbor independently with
        this probability.
    gamma : float
        Recovery probability per infected individual per time step, in range [0, 1].
        Higher values lead to shorter infection duration and lower endemic prevalence.
        The ratio beta/gamma (basic reproduction number R₀ for mean-field approximation)
        determines whether the epidemic persists.
    infected : set
        Set of node IDs that are currently infected at the start of this time step.
        This set is not modified in-place; an updated set is returned.
    
    Returns:
    - Updated set of infected nodes after this time step
    """

    new_infected = set()
    recovered = set()

    # Transmission phase: infected nodes attempt to infect susceptible neighbors
    for node in infected:
        # Iterate over neighbors that are NOT currently infected (susceptible neighbors)
        for neighbor in set(graph.neighbors(node)) - infected:
            # With probability 'beta', infect the susceptible neighbor
            if random.random() < beta:
                new_infected.add(neighbor)

    # Recovery phase: infected nodes recover with probability gamma
    for node in infected:
        # With probability 'gamma', the infected node recovers
        if random.random() < gamma:
            recovered.add(node)

    # Update infected set: add newly infected, remove recovered
    infected = infected | new_infected  # Add new infections
    infected = infected - recovered     # Remove recovered individuals

    return infected


def burn_in(graph, beta, gamma, initial_infected_count, burn_in_steps):
    """
    Run the epidemic for a burn-in period to reach a quasi-steady state.
    
    The burn-in phase allows the epidemic to evolve from arbitrary initial conditions
    to a more representative state before collecting data. This is useful for:
    - Removing transient effects from initial conditions
    - Reaching endemic equilibrium in SIS models
    - Ensuring measurements reflect typical epidemic dynamics
    
    Parameters
    ----------
    graph : NetworkX graph
        The contact network on which the epidemic spreads. Nodes represent individuals
        and edges represent contact relationships through which infection can spread.
    beta : float
        Transmission probability per contact, in range [0, 1]. At each iteration, an
        infected node attempts to infect each susceptible neighbor independently with
        this probability.
    gamma : float
        Recovery probability per infected individual per time step, in range [0, 1].
        Higher values lead to shorter infection duration and lower endemic prevalence.
        The ratio beta/gamma (basic reproduction number R₀ for mean-field approximation)
        determines whether the epidemic persists.
    initial_infected_count : int
        Number of nodes to infect at the start of the burn-in period. These are
        selected uniformly at random from all nodes in the graph.
    burn_in_steps : int
        Number of time steps to run before returning the epidemic state.
    
    Returns
    -------
    set
        Set of infected node IDs after the burn-in period. This represents a sample
        from the quasi-stationary distribution of the SIS process. The set may be
        empty if the epidemic died out during burn-in (possible for R₀ < 1 or by
        chance for small populations).
    """

    # Randomly select the initial infected nodes from the graph
    initial_infected = random.sample(list(graph.nodes()), k=initial_infected_count)
    infected = set(initial_infected)

    # Run the epidemic for the specified number of burn-in steps
    for _ in range(burn_in_steps):
        infected = epidemic_step(graph, beta, gamma, infected)
    
    return infected


def collect_snapshots(graph, beta, gamma, infected, num_snapshots, step_between):
    """
    Record epidemic state at regular time intervals for analysis.
    
    Continues an SIS epidemic simulation from a given state and captures snapshots
    of the infected population at specified intervals. Used for generating
    multiple independent observations from the endemic state.

    Parameters
    ----------
    graph : NetworkX graph
        The contact network on which the epidemic spreads. Nodes represent individuals
        and edges represent contact relationships through which infection can spread.
    beta : float
        Transmission probability per contact, in range [0, 1]. At each iteration, an
        infected node attempts to infect each susceptible neighbor independently with
        this probability.
    gamma : float
        Recovery probability per infected individual per time step, in range [0, 1].
        Higher values lead to shorter infection duration and lower endemic prevalence.
        The ratio beta/gamma (basic reproduction number R₀ for mean-field approximation)
        determines whether the epidemic persists.
    infected : set
        Initial set of infected node IDs from which to continue the simulation.
        Typically the output of burn_in() to ensure starting from quasi-equilibrium.
        This set is not modified in-place.
    num_snapshots : int
        Number of snapshots to collect. Each snapshot records one observation of
        the epidemic state.
    step_between : int
        Number of time steps to advance between consecutive snapshots. Controls
        temporal resolution and correlation between snapshots:
        - Small values: More frequent snapshots, higher temporal correlation
        - Large values: More independent snapshots, lower temporal correlation    
    
    Returns
    -------
    list of set
        List of length num_snapshots, where each element is a set of infected node
        IDs at one snapshot time. Snapshots are ordered chronologically. Each set
        is an independent object.
    """

    snapshots = []

    # Collect the specified number of snapshots
    for _ in range(num_snapshots):
        # Advance the epidemic for 'step_between' time steps
        for _ in range(step_between):
            infected = epidemic_step(graph, beta, gamma, infected)

        # Record the current state of infected nodes
        snapshots.append(set(infected))

    return snapshots