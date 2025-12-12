import numpy as np
import itertools
from scipy.integrate import solve_ivp
from scipy.special import comb, logsumexp
import matplotlib.pyplot as plt
import math
from itertools import combinations_with_replacement
from tqdm.contrib.concurrent import process_map
import pickle

N = 1
k = 100
num_genomes = 8
trials = 1000
# mu_grid_ini = np.linspace(0.0, 0.1, 100)
# mu_grid_mid = np.linspace(0.1, 0.995, 100)
# mu_grid_end = np.linspace(0.995, 1.0, 50)
# mu_grid = np.unique(np.concatenate((mu_grid_ini, mu_grid_mid, mu_grid_end)))
mu_grid = np.linspace(0.0, 1.0, 100)

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


def make_ode(mu, dt_record, t_eval_final, N, num_genomes, combos_list, combo_indices, coeff_array_base, binom_cache):

    def ode(t, x):
        x = np.maximum(x, 0)
        
        with np.errstate(divide='ignore'):
            y = np.log(x)
        
        y_N = y[combo_indices]
        
        log_prods = np.sum(y_N, axis=1)
        mask = ~np.isneginf(log_prods)

        if not np.any(mask):
            return np.zeros_like(x)

        log_prods_masked = log_prods[mask]
        coeff_array = coeff_array_base[mask]
        combos_masked = combo_indices[mask]
        
        with np.errstate(divide='ignore'):
            log_w = np.log(coeff_array) + log_prods_masked

        sums_of_a = np.sum(combos_masked, axis=1)
        q_vals = sums_of_a / (N * num_genomes) * (1 - mu)
        
        unique_q, inverse_indices = np.unique(q_vals, return_inverse=True)
        i_vals = np.arange(num_genomes + 1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p_q_part = i_vals * np.log(unique_q[:, np.newaxis])
            log_p_1mq_part = (num_genomes - i_vals) * np.log(1 - unique_q[:, np.newaxis])
            
        log_p_without_binom = np.nan_to_num(log_p_q_part + log_p_1mq_part, nan=0.0)

        unique_log_p_vecs = np.log(binom_cache) + log_p_without_binom

        log_p_vecs = unique_log_p_vecs[inverse_indices]

        log_G = logsumexp(log_w[:, None] + log_p_vecs, axis=0)
        
        total = np.sum(x)
        d_t = 0.0
        if total > 0:
            log_sum_p_vecs = logsumexp(log_p_vecs, axis=1)
            log_d_t_numer = logsumexp(log_w + log_sum_p_vecs)
            d_t = np.exp(log_d_t_numer) / total

        if np.isclose(t, t_eval_final):
            dt_record[mu] = d_t
        
        G = np.exp(log_G)
        dxdt = G - d_t * x

        return dxdt
    
    return ode

def simulate_mu(mu_val):
    
    try:
        x0 = np.zeros(num_genomes + 1)
        x0[8] = 100

        local_dt_record = {}
        if mu_val < 0.50:
            t_span_mu = (0, 1e4)
            t_eval_mu = [1e4]
        elif mu_val < 0.95:
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
    plt.savefig(f"{N}p_mu_vs_pc_log.png", format="png", dpi=300)

    plt.figure(figsize=(10, 6))
    plt.plot(mu_grid, dt_values,label="Without Complimentation", linestyle='-') 
    if mu_grid_1p is not None and dt_values_1p is not None:
        plt.plot(mu_grid_1p, dt_values_1p, label="Without Complimentation", linestyle='--')
    plt.xlabel("Mutation rate (µ)")
    plt.ylabel("d(t)")
    plt.xlim(0, 1)
    plt.title(f"μ vs. d(t) (N={N}, k={k})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{N}p_mu_vs_dt_log.png", format="png", dpi=300)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_grid, dt_values, label="Without Complimentation", linestyle='-') 
    if mu_grid_1p is not None and dt_values_1p is not None:
        plt.plot(mu_grid_1p, dt_values_1p, label="Without Complimentation", linestyle='--')
    plt.xlabel("Mutation rate (µ)")
    plt.ylabel("d(t)")
    # plt.xlim(0.95, 1)
    plt.ylim(0, 2)
    plt.title(f"μ vs. d(t) (N={N}, k={k})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{N}p_mu_vs_dt_log_zoom.png", format="png", dpi=300)
    
    dt_output_filename = f"dt_vs_mu_N{N}.pkl"
    
    try:
        with open(dt_output_filename, "rb") as f:
            old_data_dt = pickle.load(f)
        print(f"Loaded existing data from '{dt_output_filename}'.")
    except FileNotFoundError:
        old_data_dt = {'mu_grid': np.array([]), 'dt_values': np.array([])}
        print(f"'{dt_output_filename}' not found. A new file will be created.")
    
    merged_data_dict_dt = dict(zip(old_data_dt['mu_grid'], old_data_dt['dt_values']))
    new_data_dict_dt = dict(zip(mu_grid, dt_values))
    merged_data_dict_dt.update(new_data_dict_dt)
    
    sorted_mu_dt = np.array(sorted(merged_data_dict_dt.keys()))
    sorted_dt = np.array([merged_data_dict_dt[mu] for mu in sorted_mu_dt])
    
    data_to_save_dt = {
        'mu_grid': sorted_mu_dt,
        'dt_values': sorted_dt
    }
    
    with open(dt_output_filename, "wb") as f:
        pickle.dump(data_to_save_dt, f)
    print(f"Updated dt data has been saved to '{dt_output_filename}'.")
    print(f"Total dt data points: {len(sorted_mu_dt)}")


    pop_output_filename = f"pc_vs_mu_N{N}.pkl"

    try:
        with open(pop_output_filename, "rb") as f:
            old_data_pop = pickle.load(f)
        print(f"Loaded existing data from '{pop_output_filename}'.")
    except FileNotFoundError:
        old_data_pop = {'mu_grid': np.array([]), 'final_pop': np.empty((0, num_genomes + 1))}
        print(f"'{pop_output_filename}' not found. A new file will be created.")

    merged_data_dict_pop = dict(zip(old_data_pop['mu_grid'], old_data_pop['final_pop']))
    new_data_dict_pop = dict(zip(mu_grid, final_pop))
    merged_data_dict_pop.update(new_data_dict_pop)

    sorted_mu_pop = np.array(sorted(merged_data_dict_pop.keys()))
    sorted_pop = np.array([merged_data_dict_pop[mu] for mu in sorted_mu_pop])

    data_to_save_pop = {
        'mu_grid': sorted_mu_pop,
        'final_pop': sorted_pop
    }

    with open(pop_output_filename, "wb") as f:
        pickle.dump(data_to_save_pop, f)
    print(f"Updated particle counts data has been saved to '{pop_output_filename}'.")
    print(f"Total particle counts data points: {len(sorted_mu_pop)}")
    
    print("Done.")
