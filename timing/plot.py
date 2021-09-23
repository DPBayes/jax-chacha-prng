import numpy as np
import matplotlib.pyplot as plt
import os.path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=10000000)
parser.add_argument("--output-path", type=str, default="./")
args = parser.parse_args()

results = {
    'builtin': { 'random_bits': { }, 'uniform' : {} },
    'chacha': { 'random_bits': { }, 'uniform' : {} }
}

for generator, value in results.items():
    for method, results_per_sample_site in value.items():
        for sample_bits in [32, 64]:
            filename = os.path.join(
                args.output_path, f'timings_{generator}_{method}_{args.num_samples}_{sample_bits}.np')

            with open(filename, 'rb') as f:
                raw_values = np.load(f)

            raw_values *= 1000 # convert to ms
            mean = np.mean(raw_values)
            stderr = np.std(raw_values) / np.sqrt(len(raw_values))
            results_per_sample_site[sample_bits] = {'raw': raw_values, 'avg': mean, 'err': stderr}

print(results)


def subselect(generator, bits, stat):
    return [results[generator]['random_bits'][bits][stat], results[generator]['uniform'][bits][stat]]

methods = ['random_bits', 'uniform']
ticks = np.arange(len(methods))
w = .2
plt.figure()
plt.bar(ticks-1.5*w, subselect('builtin', 32, 'avg'), yerr=subselect('builtin', 32, 'err'), color='tab:blue', label='builtin (32 bit)', width=w)
plt.bar(ticks-0.5*w, subselect('builtin', 64, 'avg'), yerr=subselect('builtin', 64, 'err'), color='tab:blue', hatch='///', label='builtin (64 bit)', width=w)
plt.bar(ticks+0.5*w, subselect('chacha', 32, 'avg'), yerr=subselect('chacha', 32, 'err'), color='tab:orange', label='chacha (32 bit)', width=w)
plt.bar(ticks+1.5*w, subselect('chacha', 64, 'avg'), yerr=subselect('chacha', 64, 'err'), color='tab:orange', hatch='///', label='chacha (64 bit)', width=w)
plt.xticks(ticks, labels=methods)
plt.legend()
plt.ylabel("time [ms]")
plt.title(f"Time for ${args.num_samples}$ samples.")

plt.show()
