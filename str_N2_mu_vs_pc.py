import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

N = 2
k = 100
num_genomes = 8
t1 = 1e3
mu_ini = np.linspace(0, 0.1, 50)
mu_end = np.linspace(0.1, 1, 90)
mu_values = np.concatenate([mu_ini, mu_end])

bit_repr = np.array([[int(b) for b in format(i, '08b')] for i in range(2 ** num_genomes)], dtype=np.uint8)

hamming_weight = np.sum(bit_repr, axis=1)

def p_matrix(mu):
    genotypes = 2 ** num_genomes
    gene_seq = np.arange(genotypes, dtype=np.uint16)

    a_indices, b_indices = np.triu_indices(genotypes)

    i = gene_seq

    bits = np.arange(num_genomes - 1, -1, -1, dtype=np.uint8)
    a_bits_full = ((a_indices[:, np.newaxis] >> bits) & 1).astype(np.float64)
    b_bits_full = ((b_indices[:, np.newaxis] >> bits) & 1).astype(np.float64)

    V_eff_bits = np.where(a_bits_full == b_bits_full, a_bits_full, 0.5)
    V_eff_full = np.min(V_eff_bits, axis=1)

    mask = V_eff_full > 0

    a_active = a_indices[mask]
    b_active = b_indices[mask]
    V_eff = V_eff_full[mask]

    a_bits = a_bits_full[mask]
    b_bits = b_bits_full[mask]
    i_bits = ((i[:, np.newaxis] >> bits) & 1).astype(np.float64)

    p1 = 0.5 * (a_bits + b_bits) * (1.0 - mu)
    p0 = 1.0 - p1

    p1_tensor = p1[:, np.newaxis, :]
    p0_tensor = p0[:, np.newaxis, :]
    i_tensor = i_bits[np.newaxis, :, :]

    pi_tensor = (p1_tensor**i_tensor) * (p0_tensor**(1.0 - i_tensor))

    p = np.prod(pi_tensor, axis=2)

    factor = np.ones_like(a_active, dtype=np.float64)
    factor[a_active != b_active] = 2.0

    return a_active, b_active, i, p * V_eff[:, np.newaxis], factor

def dxdt(t, x, a, b, i, p, factor, d_log):
    dx = np.zeros_like(x)
    
    xa = x[a]
    xb = x[b]
    
    contribution_rate = k * factor * xa * xb
    
    total_contrib_matrix = p * contribution_rate[:, np.newaxis]
    
    total_contrib_to_i = np.sum(total_contrib_matrix, axis=0)
    
    dx[i] += total_contrib_to_i
    
    d = np.sum(dx) / np.sum(x) if np.sum(x) > 0 else 0
    d_log.append(d)
    dx -= d * x
    return dx

def solve_for_mu(mu_arg):
    if isinstance(mu_arg, tuple):
        mu = mu_arg[0]
    else:
        mu = mu_arg
        
    x0 = np.zeros(2 ** num_genomes)
    x0[2 ** num_genomes - 1] = 100
    a, b, i, p, factor = p_matrix(mu)
    d_log = []
    sol = solve_ivp(lambda t, x: dxdt(t, x, a, b, i, p, factor, d_log),
                    (0, t1), x0, method='BDF', dense_output=True)
    return sol.y[:, -1], d_log[-1] if d_log else 0

if __name__ == '__main__':
    results_raw = process_map(solve_for_mu, mu_values)
    
    results = np.array([r[0] for r in results_raw])
    final_d_values = np.array([r[1] for r in results_raw])
    
    distribution_by_mu = np.zeros((len(mu_values), num_genomes + 1))
    
    for h in range(num_genomes + 1):
        indices = np.where(hamming_weight == h)[0]
        distribution_by_mu[:, h] = np.sum(results[:, indices], axis=1)
        
    plt.figure(figsize=(10, 6))
    for h in range(num_genomes + 1):
        plt.plot(mu_values, distribution_by_mu[:, h], label=f'The Number of 1s: {h}')
        
    plt.xlabel('Mutation Rate (μ)')
    plt.ylabel('Particle Count')
    plt.title(f'μ vs. Particle Count (N={N}, k={k})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('256_N2_pc.png', dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, final_d_values, label=f'N={N}')
    plt.xlabel('Mutation Rate (μ)')
    plt.ylabel('d(t)')
    plt.title(f'μ vs. Death Rate (N={N}, k={k}, t={t1})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('256_N2_dt.png', dpi=300)
    plt.show()
