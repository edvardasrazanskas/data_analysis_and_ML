import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from collections import defaultdict
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import scipy.stats as st
from scipy.optimize import minimize
import itertools

def brute_force_ground_state(h, J, offset, n_assets, bits_per_asset, cov_matrix=None):

    num_qubits = n_assets * bits_per_asset
    if num_qubits > 20:
        raise ValueError("Brute-force search is practical only for ≤ 20 qubits.")

    def energy_of(bits):
        # Convert {0,1} → {+1,-1} for Ising
        z = [1 - 2*b for b in bits]
        e = offset
        for (i,), coeff in h.items():
            e += coeff * z[i]
        for (i, j), coeff in J.items():
            e += coeff * z[i] * z[j]
        return e

    ground_energy = None
    ground_states = []

    # CORRECTED: Generate bits in proper qubit order
    for bits in itertools.product([0, 1], repeat=num_qubits):
        e = energy_of(bits)
        if ground_energy is None or e < ground_energy - 1e-12:
            ground_energy = e
            ground_states = [bits]
        elif abs(e - ground_energy) < 1e-12:
            ground_states.append(bits)

    results = []
    for bits in ground_states:
        # CORRECTED: Create bitstring to match Qiskit convention
        # Qiskit expects qubit 0 on the RIGHT, so we need to reverse the entire string
        bitstring = ''.join(str(b) for b in bits)[::-1]
        
        entry = {
            'bitstring': bitstring,
            'energy': ground_energy,
        }
        if cov_matrix is not None:
            allocs = decode_portfolio_allocation(bitstring, n_assets, bits_per_asset)
            entry['allocations'] = allocs
            entry['total_allocation'] = sum(allocs.values())
            entry['risk'] = calculate_portfolio_risk(allocs, cov_matrix)
        results.append(entry)

    return results


def qubo_equality_contraint(cov_matrix, n_assets, bits_per_asset, budget, returns, return_coeff, lambd):
    total_qubits = n_assets * bits_per_asset
    Q = np.zeros((total_qubits, total_qubits))
    
    # Correct binary weights (big-endian: MSB first)
    weights = [2**(-i-1) for i in range(bits_per_asset)]  # [0.125, 0.25, 0.5] for 3 bits
    
    # Objective: Minimize variance
    for i in range(n_assets):
        for j in range(i, n_assets):  # Avoid double counting: j >= i
            for bi in range(bits_per_asset):
                for bj in range(bits_per_asset):
                    idx_i = i * bits_per_asset + bi
                    idx_j = j * bits_per_asset + bj
                    w_i = weights[bi]
                    w_j = weights[bj]
                    
                    # Variance term (symmetric handling)
                    coeff = cov_matrix[i, j] * w_i * w_j
                    if idx_i == idx_j:
                        Q[idx_i, idx_j] += coeff
                    elif idx_i < idx_j:
                        # FIXED: Put full coefficient in upper triangular part
                        if i == j:  # Same asset, different bits
                            Q[idx_i, idx_j] += 2 * coeff  # Factor of 2 for symmetric expansion
                        else:  # Different assets
                            Q[idx_i, idx_j] += 2 * coeff  # Factor of 2 for symmetric expansion
    
    # Budget constraint: λ*(sum w_i x_i - budget)^2
    # Quadratic part: λ * w_i * w_j
    for i in range(n_assets):
        for j in range(i, n_assets):  # j >= i
            for bi in range(bits_per_asset):
                for bj in range(bits_per_asset):
                    idx_i = i * bits_per_asset + bi
                    idx_j = j * bits_per_asset + bj
                    w_i = weights[bi]
                    w_j = weights[bj]
                    
                    # Budget-constraint (Σ w_i x_i − B)² expands to
                    #   λ w_i² x_i               (diagonal)
                    # + 2 λ w_i w_j x_i x_j      (i ≠ j)
                    # For the diagonal entries we therefore add the
                    # *single* coefficient  λ w_i²

                    if idx_i == idx_j:
                        Q[idx_i, idx_j] += lambd * w_i * w_j
                    elif idx_i < idx_j:
                        # Full coefficient for upper triangular
                        Q[idx_i, idx_j] += 2 * lambd * w_i * w_j
    
    # Linear parts of budget constraint and returns
    for i in range(n_assets):
        for bi in range(bits_per_asset):
            idx = i * bits_per_asset + bi
            w_i = weights[bi]
            
            # Budget linear term: -2λ * budget * w_i
            Q[idx, idx] += -2 * lambd * budget * w_i
            
            # Returns: subtract for maximization
            Q[idx, idx] -= return_coeff * returns[i] * w_i
    
    return Q

def qubo_unbalanced_penalty(cov_matrix, n_assets, bits_per_asset, budget, returns, return_coeff, lambd_under, lambd_over):
    total_qubits = n_assets * bits_per_asset
    Q = np.zeros((total_qubits, total_qubits))
    
    weights = [2**(-i-1) for i in range(bits_per_asset)]
    
    # 1. Risk minimization (same as equality constraint)
    for i in range(n_assets):
        for j in range(i, n_assets):
            for bi in range(bits_per_asset):
                for bj in range(bits_per_asset):
                    idx_i = i * bits_per_asset + bi
                    idx_j = j * bits_per_asset + bj
                    w_i, w_j = weights[bi], weights[bj]
                    
                    coeff = cov_matrix[i, j] * w_i * w_j
                    if idx_i == idx_j:
                        Q[idx_i, idx_j] += coeff
                    elif idx_i < idx_j:
                        # Full coefficient in upper triangular
                        Q[idx_i, idx_j] += 2 * coeff
    
    # 2. Unbalanced penalization
    # Use NET penalty coefficient: (λ_over - λ_under)
    net_penalty_coeff = lambd_over - lambd_under
    
    for i in range(n_assets):
        for j in range(i, n_assets):
            for bi in range(bits_per_asset):
                for bj in range(bits_per_asset):
                    idx_i = i * bits_per_asset + bi
                    idx_j = j * bits_per_asset + bj
                    w_i, w_j = weights[bi], weights[bj]
                    
                    if idx_i == idx_j:
                        Q[idx_i, idx_j] += net_penalty_coeff * w_i * w_j
                    elif idx_i < idx_j:
                        # Full coefficient in upper triangular
                        Q[idx_i, idx_j] += 2 * net_penalty_coeff * w_i * w_j
    
    # 3. Linear terms
    for i in range(n_assets):
        for bi in range(bits_per_asset):
            idx = i * bits_per_asset + bi
            w_i = weights[bi]
            
            # Linear term: 2(λ_under - λ_over) * budget * w_i
            Q[idx, idx] += 2 * (lambd_under - lambd_over) * budget * w_i
            
            # Returns maximization
            Q[idx, idx] -= return_coeff * returns[i] * w_i
    
    return Q


def QUBO_to_Ising(Q, offset):
    """    
    Parameters:
    - Q: QUBO matrix
    - offset: Initial offset value
    
    Returns:
    - h: Dictionary of single-qubit terms
    - J: Dictionary of two-qubit interaction terms
    - offset: Updated offset value
    """
    n_qubits = len(Q)
    h = defaultdict(int)
    J = defaultdict(int)
    
    for i in range(n_qubits):
        # diagonal terms
        h[(i,)] -= Q[i, i] / 2
        offset += Q[i, i] / 2

        # off-diagonal terms (j > i)
        for j in range(i + 1, n_qubits):
            J[(i, j)] += Q[i, j] / 4
            h[(i,)] -= Q[i, j] / 4
            h[(j,)] -= Q[i, j] / 4
            offset += Q[i, j] / 4
            
    return h, J, offset

def qaoa_circuit(gammas, betas, h, J, num_qubits):
    """
    Parameters:
    - gammas: Parameters for the cost Hamiltonian
    - betas: Parameters for the mixer Hamiltonian
    - h: Dictionary of single-qubit terms
    - J: Dictionary of two-qubit interaction terms
    - num_qubits: Number of qubits in the circuit
    
    Returns:
    - qc: Quantum circuit implementing QAOA
    """
    p = len(gammas)
    # --- Optional coefficient normalisation ---
    # Previous version divided every coefficient by the largest |h| or |J|,
    # which cancelled any attempt to strengthen the budget-constraint by
    # multiplying it with a large λ.  The following line is kept for
    # reference but is **NOT** used in the angle computation any more.  If you
    # ever want the old behaviour back just divide by `wmax` again.
    wmax = max(
        np.max(np.abs(list(h.values()))) if h else 0,
        np.max(np.abs(list(J.values()))) if J else 0,
    )

    # Create a quantum circuit with num_qubits qubits
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Apply the initial layer of Hadamard gates to all qubits
    for i in range(num_qubits):
        qc.h(i)
    
    # Repeat p layers of the QAOA circuit
    for layer in range(p):
        # ---------- COST HAMILTONIAN ----------
        for ki, v in h.items():  # single-qubit terms
            qc.rz(2 * gammas[layer] * v / wmax, ki[0])
        
        for kij, vij in J.items():  # two-qubit terms
            qc.cx(kij[0], kij[1])
            qc.rz(2 * gammas[layer] * vij / wmax, kij[1])
            qc.cx(kij[0], kij[1])
        
        # ---------- MIXER HAMILTONIAN ----------
        for i in range(num_qubits):
            qc.rx(-2 * betas[layer], i)
    
    # Measure all qubits
    qc.measure(range(num_qubits), range(num_qubits))
    
    return qc

def qaoa_expectation(params, h, J, num_qubits, shots=5000):
    """
    Compute the expectation value of the cost Hamiltonian for given QAOA parameters.
    
    Args:
        params: Array of QAOA parameters (gammas followed by betas)
        h: Dictionary of single-qubit terms
        J: Dictionary of two-qubit interaction terms
        num_qubits: Number of qubits
        n_assets: Number of assets
        bits_per_asset: Number of bits per asset
        cov_matrix: Covariance matrix of assets
        shots: Number of circuit measurements
    
    Returns:
        exp_val: Expectation value (average energy)
    """
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]
    
    # Create and run the QAOA circuit
    qc = qaoa_circuit(gammas, betas, h, J, num_qubits)
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    
    # Compute expectation value
    exp_val = 0
    total_shots = 0
    
    for bitstring, count in counts.items():
        z = np.array([1 - 2*int(b) for b in bitstring[::-1]])

        # energy of H = Σ h_i z_i  +  Σ_{i<j} J_{ij} z_i z_j
        energy = 0.0
        for (i,), coeff in h.items():
            energy += coeff * z[i]
        for (i, j), coeff in J.items():
            energy += coeff * z[i] * z[j]
    
        exp_val += energy * count
        total_shots += count
    
    return exp_val / total_shots

def optimize_qaoa_parameters(h, J, num_qubits, p, shots=5000, maxiter=100):
    """
    Optimize QAOA parameters using COBYLA.
    
    Args:
        h: Dictionary of single-qubit terms
        J: Dictionary of two-qubit interaction terms
        num_qubits: Number of qubits
        p: Number of QAOA layers
        shots: Number of circuit measurements
        maxiter: Maximum number of iterations
    
    Returns:
        opt_result: Optimization result object
    """
    # Initial parameters (as used in your original code)
    initial_gammas = np.linspace(0.1, 1.0, p)
    initial_betas = np.linspace(0.1, 0.8, p)[::-1]
    initial_params = np.concatenate([initial_gammas, initial_betas])
    
    # Optimize using COBYLA
    result = minimize(
        qaoa_expectation,
        initial_params,
        args=(h, J, num_qubits, shots),
        method='COBYLA',
        options={
            'maxiter': maxiter,
            'rhobeg': 0.1,  # Initial trust region radius
            'disp': True    # Display optimization progress
        }
    )
    
    return result

def decode_portfolio_allocation(bitstring, n_assets, bits_per_asset):
    """    
    Parameters:
    - bitstring: Binary string from measurement
    - n_assets: Number of assets
    - bits_per_asset: Number of binary variables per asset
    
    Returns:
    - allocations: Dictionary mapping asset indices to allocation percentages
    """
    # Qiskit represents measurement strings with qubit-0 in the **right-most**
    # position.  To map them back to the order used when the QUBO and circuit
    # were built (qubit-0 is the **first** weight bit of asset 0) we must first
    # reverse the string so that index 0 corresponds to qubit 0.

    bitstring = bitstring[::-1]

    allocations = {}
    weights = [2**(-i-1) for i in range(bits_per_asset)]  # MSB first

    for i in range(n_assets):
        start = i * bits_per_asset
        asset_bits = bitstring[start : start + bits_per_asset]

        allocation = sum(int(bit) * weight for bit, weight in zip(asset_bits, weights))
        allocations[i] = allocation

    return allocations

def calculate_portfolio_risk(allocations, cov_matrix):
    """    
    Parameters:
    - allocations: Dictionary mapping asset indices to allocation percentages
    - cov_matrix: Covariance matrix of assets
    
    Returns:
    - risk: Portfolio risk (variance)
    """
    alloc_vector = np.zeros(len(cov_matrix))
    for idx, alloc in allocations.items():
        alloc_vector[idx] = alloc
    
    risk = alloc_vector.T @ cov_matrix @ alloc_vector
    return risk

def valid_top_portfolios_infos(counts, n_assets, bits_per_asset, cov_matrix, tickers, budget=1.0, top_n=3, tolerance=0.05):
    """
    Returns:
    - top_portfolios: List of dictionaries containing portfolio information
    """
    # Store information about all valid portfolios
    all_portfolios = []
    
    for bitstring, count in counts.items():
        # Decode the bitstring into portfolio allocations
        allocations = decode_portfolio_allocation(bitstring, n_assets, bits_per_asset)
        
        # Calculate the total allocation
        total_allocation = sum(allocations.values())
        
        # Check if the allocation is close to the budget constraint
        if abs(total_allocation - budget) < tolerance:  # Allow some tolerance (±10%)
            # Calculate portfolio risk
            risk = calculate_portfolio_risk(allocations, cov_matrix)
            
            # Create a dictionary with portfolio information
            portfolio_info = {
                'bitstring': bitstring,
                'risk': risk,
                'volatility': np.sqrt(risk),
                'allocations': allocations,
                'total_allocation': total_allocation,
                'count': count  # How many times this solution appeared
            }
            
            all_portfolios.append(portfolio_info)
    
    # Sort portfolios by risk (ascending)
    all_portfolios.sort(key=lambda x: x['count'], reverse=True)
    
    # Take the top N portfolios
    top_portfolios = all_portfolios[:top_n]
    
    return top_portfolios

def top_portfolios_by_energy(counts, h, J, offset, n_assets, bits_per_asset, cov_matrix, tickers, top_n=3):
    """Return *top_n* bit-strings with the lowest Ising energy.

    Parameters
    ----------
    counts : dict
        Measurement histogram {bitstring: shots}.
    h, J, offset : dict, dict, float
        Ising coefficients as produced by ``QUBO_to_Ising``.
    n_assets, bits_per_asset : int
        Encoding parameters (for decoding allocations).
    cov_matrix : ndarray
        Covariance matrix used to compute portfolio risk.
    tickers : list[str]
        Names of the assets – only used for readability in the output.
    top_n : int, default 3
        How many portfolios to return.

    Returns
    -------
    list[dict]
        Each dictionary contains the same keys as in
        :func:`valid_top_portfolios_infos` **plus** the entry ``'energy'``.
    """
    portfolios = []

    for bitstring, shot_count in counts.items():
        # --- decode & derived classical quantities -------------------
        allocs = decode_portfolio_allocation(bitstring, n_assets, bits_per_asset)
        total_alloc = sum(allocs.values())
        risk = calculate_portfolio_risk(allocs, cov_matrix)

        # --- Ising energy  E = offset + Σ h_i z_i + Σ J_ij z_i z_j ----
        z = [1 - 2*int(b) for b in bitstring[::-1]]
        energy = offset
        for (i,), coeff in h.items():
            energy += coeff * z[i]
        for (i, j), coeff in J.items():
            energy += coeff * z[i] * z[j]

        portfolios.append({
            'bitstring': bitstring,
            'energy': energy,
            'risk': risk,
            'volatility': np.sqrt(risk),
            'allocations': allocs,
            'total_allocation': total_alloc,
            'count': shot_count,
        })

    portfolios.sort(key=lambda x: x['energy'])
    return portfolios[:top_n] 
