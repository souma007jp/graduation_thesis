import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import comb
import math
import time
from itertools import combinations_with_replacement
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import sys

mu = 0.1
k = 100
num_genomes = 8
N = 9
trials = 1000
t_span = (0, 1e-8)
t_eval = np.linspace(0, 1e-8, 100)

veff_file = f"veff_lookup_N{N}_T{trials}.pkl"
try:
    with open(veff_file, "rb") as f:
        V_eff_lookup = pickle.load(f)
    print(f"Successfully loaded V_eff_lookup from '{veff_file}'")
except FileNotFoundError:
    print(f"Error: '{veff_file}' not found.")
    print("Please run montecarlo.py first to generate the V_eff lookup table.")
    sys.exit(1)


combos_list = list(combinations_with_replacement(range(num_genomes + 1), N))
combo_indices = np.array(combos_list)

sym_factors_array = np.array([
    math.factorial(N) // np.prod([math.factorial(c) for c in np.unique(combo, return_counts=True)[1]])
    for combo in combos_list
])

veff_array_full = np.array([V_eff_lookup.get(c, 0.0) for c in combos_list])

binom_cache = comb(num_genomes, np.arange(num_genomes + 1))
p_vec_cache = {}

def get_p_vec(q):
    q_rounded = round(q, 4)
    if q_rounded in p_vec_cache:
        return p_vec_cache[q_rounded]
    i_vals = np.arange(num_genomes + 1)
    p_vec = binom_cache * q_rounded**i_vals * (1 - q_rounded)**(num_genomes - i_vals)
    p_vec_cache[q_rounded] = p_vec
    return p_vec

def q_func_vec(a_vals_matrix, mu):
   return np.sum(a_vals_matrix, axis=1) / (N * num_genomes) * (1 - mu)

def make_ode(mu, V_eff_lookup):
    def ode(t, x):
        dxdt = np.zeros_like(x)

        x_N = x[combo_indices]
        prods = np.prod(x_N, axis=1)
        mask = prods > 0
        
        if not np.any(mask):
            return dxdt

        combos_masked = combo_indices[mask]
        prods_masked = prods[mask]
        sym_factors_masked = sym_factors_array[mask]
        veff_array_masked = veff_array_full[mask]

        q_vals = q_func_vec(combos_masked, mu)

        unique_q, q_inverse = np.unique(q_vals, return_inverse=True)
        p_vecs_unique = np.stack([get_p_vec(q) for q in unique_q])
        p_vecs = p_vecs_unique[q_inverse]

        coeffs = sym_factors_masked * k * veff_array_masked
        w = coeffs * prods_masked

        dxdt = np.sum(w[:, None] * p_vecs, axis=0)

        total = np.sum(x)
        if total > 0:
            d_t = np.sum(w) / total
        else:
            d_t = 0.0

        dxdt -= d_t * x
        return dxdt
    return ode

x0 = np.zeros(num_genomes + 1)
x0[8] = 100

start_time = time.time()

sol = solve_ivp(
    make_ode(mu, V_eff_lookup),
    t_span,
    x0,
    t_eval=t_eval,
    method='BDF',
    atol=1e-6,
    rtol=1e-8,
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"微分方程式の計算時間: {elapsed_time:.2f} 秒")

plt.figure(figsize=(10, 6))
for i in range(num_genomes + 1):
    plt.plot(sol.t, sol.y[i], label=f"x_{i}")
plt.xlabel("Time t")
plt.ylabel("Particle Count")
plt.title(f"Time Evolution of Particle Count (N={N}, μ={mu})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()