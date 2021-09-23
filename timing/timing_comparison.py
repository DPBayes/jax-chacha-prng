from jax.config import config
config.update("jax_enable_x64", True)
import jax.random
import jax._src.random
import chacha.random
import time
import numpy as np
from tqdm import tqdm
import os.path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("generator", choices=['builtin', 'chacha'])
parser.add_argument("method", choices=['random_bits', 'uniform'])
parser.add_argument("--num_samples", type=int, default=10000000)
parser.add_argument("--runs", type=int, default=100)
parser.add_argument("--sample-bits", type=int, choices=[32, 64], default=32)
parser.add_argument("--output-path", type=str, default="./")
args = parser.parse_args()

float_type = np.float32 if args.sample_bits == 32 else np.float64

if args.generator == 'builtin':
    PRNGKey = jax.random.PRNGKey
    split = jax.random.split
    if args.method == 'random_bits':
        generator_fn = lambda rng_key: jax._src.random._random_bits(rng_key, args.sample_bits, (args.num_samples,))
    elif args.method == 'uniform':
        generator_fn = lambda rng_key: jax.random.uniform(rng_key, (args.num_samples,), dtype=float_type)
elif args.generator == 'chacha':
    PRNGKey = chacha.random.PRNGKey
    split = chacha.random.split
    if args.method == 'random_bits':
        generator_fn = lambda rng_key: chacha.random.random_bits(rng_key, args.sample_bits, (args.num_samples,))
    elif args.method == 'uniform':
        generator_fn = lambda rng_key: chacha.random.uniform(rng_key, (args.num_samples,), dtype=float_type)

key = PRNGKey(0)
run_keys = split(key, args.runs)
times = np.empty(len(run_keys))

generator_fn(key)
for i, rng_key in tqdm(enumerate(run_keys)):
    start_time = time.time()
    generator_fn(rng_key)
    stop_time = time.time()
    times[i] = stop_time - start_time

filename = os.path.join(args.output_path, f'timings_{args.generator}_{args.method}_{args.num_samples}_{args.sample_bits}.np')
with open(filename, "wb") as f:
    np.save(f, times)