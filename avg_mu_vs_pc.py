import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import comb
import matplotlib.pyplot as plt
import math
from itertools import combinations_with_replacement
from tqdm.contrib.concurrent import process_map
import pickle

N = 2
k = 100
num_genomes = 8
trials = 10000
mu_grid = np.linspace(0.0, 1.0, 200)

file_name = f"veff_lookup_N{N}_T{trials}.pkl"
try:
    with open(file_name, "rb") as f:
        V_eff_lookup = pickle.load(f)

except FileNotFoundError:
    print(f"Error: The precomputed file '{file_name}' was not found.")
    print("Please run the Monte Carlo calculation script first.")
    exit()

combos_list = list(combinations_with_replacement(range(num_genomes + 1), N))
combo_indices = np.array(combos_list)

sym_factors_array = np.array([
    math.factorial(N) // np.prod([math.factorial(c) for c in np.unique(combo, return_counts=True)[1]])
    for combo in combos_list
])

veff_array = np.array([V_eff_lookup.get(combo, 0.0) for combo in combos_list])
coeff_array_base = sym_factors_array * k * veff_array
binom_cache = comb(num_genomes, np.arange(num_genomes + 1))

def make_ode(mu, dt_record, t_eval_final, N, num_genomes, combo_list, combo_indices, coeff_array_base, binom_cache):

    i_vals = np.arange(num_genomes + 1)

    def ode(t, x):
        dxdt = np.zeros_like(x)
        x_N = x[combo_indices]
        prods = np.prod(x_N, axis=1)
        mask = prods > 0
        
        if not np.any(mask):
            return np.zeros_like(x)

        prods = prods[mask]
        coeff_array = coeff_array_base[mask]
        combos_masked = combo_indices[mask]
        
        sums_of_a = np.sum(combos_masked, axis=1)
        q_vals = sums_of_a / (N * num_genomes) * (1 - mu)

        q_vals_col = q_vals[:, np.newaxis]
        p_vecs = binom_cache * (q_vals_col ** i_vals) * ((1 - q_vals_col) ** (num_genomes - i_vals))

        w = coeff_array * prods
        dxdt = np.sum(w[:, np.newaxis] * p_vecs, axis=0)
        total = np.sum(x)
        
        if total > 0:
            d_t_numer = np.sum(w * np.sum(p_vecs, axis=1))
            d_t = d_t_numer / total
        else:
            d_t = 0.0
        
        if np.isclose(t, t_eval_final):
            dt_record[mu] = d_t
            
        dxdt -= d_t * x
        return dxdt
    return ode

def simulate_mu(mu_val):
    
    try:
        x0 = np.zeros(num_genomes + 1)
        x0[8] = 100
        local_dt_record = {}
        if mu_val < 0.95:
            t_span_mu = (0, 1e4)
            t_eval_mu = [1e4]
        else:
            t_span_mu = (0, 1e4)
            t_eval_mu = [1e4]

        sol = solve_ivp(
            make_ode(mu_val, local_dt_record, t_eval_mu[-1], N, num_genomes, combos_list, combo_indices, coeff_array_base, binom_cache),
            t_span_mu,
            x0,
            t_eval=t_eval_mu,
            method='BDF',
            atol=1e-6,
            rtol=1e-8,
        )
        
        if sol.success:
            final = sol.y[:, -1]
        else:
            final = np.full(num_genomes + 1, np.nan)
        
        dt_val = local_dt_record.get(mu_val, np.nan)
        return (mu_val, final, dt_val)
    except Exception as e:
        print(f"[Error] μ={mu_val:.4f}: {e}")
        return (mu_val, np.full(num_genomes + 1, np.nan), np.nan)


if __name__ == "__main__":
    try:
        with open("dt_vs_mu_N1.pkl", "rb") as f:
            data_1p = pickle.load(f)
        mu_grid_1p = data_1p['mu_grid']
        dt_values_1p = data_1p['dt_values']
        print("'dt_vs_mu_N1.pkl' has been loaded successfully.")
    except FileNotFoundError:
        print("Error: The precomputed file 'dt_vs_mu_N1.pkl' was not found.")
        mu_grid_1p = None
        dt_values_1p = None
    
    print(f"'{file_name}' has been loaded successfully.")
    
    print(f"Starting parallel simulation for {len(mu_grid)} μ values...")
    results = process_map(simulate_mu, mu_grid, max_workers=None, chunksize=1)

    final_pop = np.array([r[1] for r in results])
    dt_record = {r[0]: r[2] for r in results}
    dt_values = np.array([dt_record.get(mu, np.nan) for mu in mu_grid])

    plt.figure(figsize=(10, 6))
    for i in range(num_genomes + 1):
        plt.plot(mu_grid, final_pop[:, i], label=f"The Number of 1s: {i}")
    plt.xlabel("Mutation rate (µ)")
    plt.ylabel("Particle Count")
    plt.title(f"μ vs. Particle Count (N={N}, k={k})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"9_N{N}.png", format="png", dpi=300)

    plt.figure(figsize=(10, 6))
    plt.plot(mu_grid, dt_values,label="With Complimentation", linestyle='-') 
    if mu_grid_1p is not None and dt_values_1p is not None:
        plt.plot(mu_grid_1p, dt_values_1p, label="Without Complimentation", linestyle='--')
    plt.xlabel("Mutation rate (µ)")
    plt.ylabel("d(t)")
    plt.xlim(0, 1)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='d(t)=1.0')
    plt.title(f"μ vs. d(t) (N={N}, k={k})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{N}p_mu_vs_dt.png", format="png", dpi=300)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_grid, dt_values, label="With Complimentation", linestyle='-') 
    if mu_grid_1p is not None and dt_values_1p is not None:
        plt.plot(mu_grid_1p, dt_values_1p, label="Without Complimentation", linestyle='--')
    plt.xlabel("Mutation rate (µ)")
    plt.ylabel("d(t)")
    plt.xlim(0, 1)
    plt.ylim(0, 2)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='d(t)=1.0')
    plt.title(f"μ vs. d(t) (k={k})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{N}p_mu_vs_dt_zoom.png", format="png", dpi=300)
    
    output_filename = f"dt_vs_mu_N{N}.pkl"
    
    try:
        with open(output_filename, "rb") as f:
            old_data = pickle.load(f)
        print(f"Loaded existing data from '{output_filename}'.")
    except FileNotFoundError:
        old_data = {'mu_grid': np.array([]), 'dt_values': np.array([])}
        print(f"'{output_filename}' not found. A new file will be created.")
    
    merged_data_dict = dict(zip(old_data['mu_grid'], old_data['dt_values']))
    
    new_data_dict = dict(zip(mu_grid, dt_values))
    merged_data_dict.update(new_data_dict)
    
    sorted_mu = np.array(sorted(merged_data_dict.keys()))
    sorted_dt = np.array([merged_data_dict[mu] for mu in sorted_mu])
    
    data_to_save = {
        'mu_grid': sorted_mu,
        'dt_values': sorted_dt
    }
    
    with open(output_filename, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Updated data has been saved to '{output_filename}'.")
    print(f"Total data points: {len(sorted_mu)}")
    
    print("Done.")
