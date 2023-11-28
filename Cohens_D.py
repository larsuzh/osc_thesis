import numpy as np
from Evaluation.mags.mixed_regular_objecto import group1a, group1b

group1a = np.array(group1a)

group1b = np.array(group1b)

def cohens_d_bootstrap(group1a, group1b, num_bootstrap_samples=1000):
    cohen_d_values = np.zeros(num_bootstrap_samples)

    n1, n2 = len(group1a), len(group1b)

    for i in range(num_bootstrap_samples):
        bootstrap_sample_1a = np.random.choice(group1a, size=n1, replace=True)
        bootstrap_sample_1b = np.random.choice(group1b, size=n2, replace=True)

        mean_1a, mean_1b = np.mean(bootstrap_sample_1a), np.mean(bootstrap_sample_1b)

        std_1a, std_1b = np.std(bootstrap_sample_1a, ddof=1), np.std(bootstrap_sample_1b, ddof=1)

        pooled_std = np.sqrt(((std_1a**2 * (n1 - 1)) + (std_1b**2 * (n2 - 1))) / (n1 + n2 - 2))

        cohen_d_values[i] = (mean_1a - mean_1b) / pooled_std

    return np.mean(cohen_d_values), np.std(cohen_d_values, ddof=1)

# Print Cohen's d values
print(cohens_d_bootstrap(group1a, group1b))