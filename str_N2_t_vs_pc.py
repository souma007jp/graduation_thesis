import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N = 2
k = 100
num_genomes = 8
t1 = 1e3
target_mu = 0.01

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

def dxdt(t, x, a, b, i, p, factor):
    dx = np.zeros_like(x)
    
    xa = x[a]
    xb = x[b]
    
    contribution_rate = k * factor * xa * xb
    
    total_contrib_matrix = p * contribution_rate[:, np.newaxis]
    
    total_contrib_to_i = np.sum(total_contrib_matrix, axis=0)
    
    dx[i] += total_contrib_to_i
    
    d = np.sum(dx) / np.sum(x) if np.sum(x) > 0 else 0
    dx -= d * x
    return dx

if __name__ == '__main__':
    x0 = np.zeros(2 ** num_genomes)
    x0[2 ** num_genomes - 1] = 100
    
    a, b, i, p, factor = p_matrix(target_mu)
    
    sol = solve_ivp(lambda t, x: dxdt(t, x, a, b, i, p, factor),
                    (0, t1), x0, method='BDF', dense_output=True)
    
    time_points = sol.t
    y_data = sol.y
    
    distribution_over_time = np.zeros((num_genomes + 1, len(time_points)))
    
    for h in range(num_genomes + 1):
        indices = np.where(hamming_weight == h)[0]
        distribution_over_time[h, :] = np.sum(y_data[indices, :], axis=0)
        
    plt.figure(figsize=(10, 6))
    for h in range(num_genomes + 1):
        plt.plot(time_points, distribution_over_time[h, :], label=f'The Number of 1s: {h}')
        
    plt.xlabel('Time (t)')
    plt.ylabel('Particle Count')
    plt.title(f'Time Evolution of Particle Count (N={N}, k={k}, μ={target_mu})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'256_t_vs_pc_N{N}_mu{target_mu}.png', dpi=300)
    plt.show()