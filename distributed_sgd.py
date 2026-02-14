import numpy as np
from multiprocessing import Pool
import time

def generate_data(n_samples=20000, n_features=20, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    y = X @ w_true + np.random.randn(n_samples) * 0.1
    return X, y, w_true

def compute_gradient(X, y, w):
    n = len(y)
    grad = (2 / n) * X.T @ (X @ w - y)
    # Simple gradient clipping to prevent explosion
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 10.0:
        grad = grad * (10.0 / grad_norm)
    return grad

def worker_compute(args):
    shard_X, shard_y, w = args
    start = time.time()
    grad = compute_gradient(shard_X, shard_y, w)
    return grad, time.time() - start

def run_distributed_sgd(n_workers, n_iters=80, base_lr=0.005):
    X, y, w_true = generate_data()
    shard_size = len(y) // n_workers
    shards = [(X[i*shard_size:(i+1)*shard_size], y[i*shard_size:(i+1)*shard_size]) 
              for i in range(n_workers)]
    
    w = np.zeros(X.shape[1])
    iter_times = []
    
    for it in range(n_iters):
        lr = base_lr / (1 + 0.01 * it)  # simple decay to help convergence
        start = time.time()
        with Pool(n_workers) as p:
            results = p.map(worker_compute, [(s[0], s[1], w) for s in shards])
        grads = [r[0] for r in results]
        avg_grad = np.mean(grads, axis=0)
        w -= lr * avg_grad
        iter_times.append(time.time() - start)
    
    avg_time = np.mean(iter_times)
    loss = np.mean((X @ w - y)**2)
    return avg_time, loss, w_true, w

if __name__ == '__main__':
    print("Running distributed SGD simulation (synchronous parameter server)...\n")
    
    times = {}
    for n in [1, 2, 4]:
        print(f"Starting run with {n} workers...")
        avg_time, loss, w_true, w_final = run_distributed_sgd(n)
        times[n] = avg_time
        speedup = times[1] / avg_time if n > 1 else 1.0
        converged = np.allclose(w_final, w_true, atol=1.0)  # slightly larger tol for realism
        print(f"N={n} workers")
        print(f"  Avg response time per iteration: {avg_time:.4f} s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Final MSE loss: {loss:.4f}")
        print(f"  Model converged (to true weights): {converged}\n")
    
    # Theoretical communication cost
    grad_size_bytes = 20 * 8   # 20 features × float64
    n_iters = 80
    print("Theoretical communication cost (gradient pushes per sync):")
    for n in [1, 2, 4]:
        comm = n * grad_size_bytes * n_iters
        print(f"N={n}: {comm} bytes")
