import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Literal
from minisom import MiniSom

def train_som(X, d1, d2, sigma, learning_rate, num_iteration, topology="rectangular", use_epochs=True, initial_weights=None, verb=False, random_seed=None):
	som = MiniSom(d1, d2, input_len=X.shape[1], sigma=sigma, learning_rate=learning_rate, topology=topology, random_seed=random_seed)

	if initial_weights is None:
		som.random_weights_init(X)
	else:
		som._weights = initial_weights

	som.train(X, num_iteration=num_iteration, use_epochs=use_epochs, verbose=verb)

	if verb:
		QE, TE, ATE = calc_som_main_qualities(som, X)
		print("\nBrief quality of SOM:")
		print(f"Quantization error:\t{QE}")
		print(f"Topographic error:\t{TE}")
		print(f"Actual topographic error:\t{ATE}")
	return som

def calc_recommended_grid_size(X):
	N = len(X)
	dd = 5 * np.sqrt(N)
	d = int(np.sqrt(dd).round())
	return dd, d

def calc_som_main_qualities(som:MiniSom, X, digits=3):
	QE = np.round(som.quantization_error(X), digits)
	TE = np.round(som.topographic_error(X), digits)
	ATE = np.round(calc_topographic_error(som, X), digits)
	return QE, TE, ATE

def calc_som_quality(som, X, round_prec = 4):
	quality_index = ["TE", "Sum of QE", "Mean QE", "Median QE", "95th percentile", "98th percentile", "99th percentile", "Max QE"]

	qes = np.linalg.norm(X - som.quantization(X), axis=1)
	qualities = np.array([
        som.topographic_error(X),
        np.sum(qes),
        np.mean(qes),
        np.median(qes),
        np.percentile(qes, 95),
        np.percentile(qes, 98),
        np.percentile(qes, 99),
        np.max(qes)
    ]).round(round_prec)

	return pd.Series(qualities, index=quality_index)

def calc_topographic_error(som:MiniSom, data, neighborhood:Literal["moore", "von-neumann"]="von-neumann", aggregated=True):
	total_neurons = np.prod(som._activation_map.shape)
	if total_neurons == 1:
		print('The topographic error is not defined for a 1-by-1 map.')
		return np.nan
	if som.topology == 'hexagonal':
		return np.nan
	else:
		return topographic_error_rectangular(som, data, neighborhood, aggregated)
	
def topographic_error_rectangular(som:MiniSom, X, neighborhood:Literal["moore", "von-neumann"]="von-neumann", aggregated=True):
        """Return the topographic error for rectangular grid"""
        t = 1.42 if neighborhood == "moore" else 1
        # b2mu: best 2 matching units
        b2mu_inds = np.argsort(som._distance_from_weights(X), axis=1)[:, :2]
        b2my_xy = np.unravel_index(b2mu_inds, som._weights.shape[:2])
        b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1]
        dxdy = np.hstack([np.diff(b2mu_x), np.diff(b2mu_y)])
        distance = np.linalg.norm(dxdy, axis=1)
        return (distance > t).mean() if aggregated is True else (distance > t)

def place_node_edges(som, ax=None):
	if ax is None:
		ax = plt.gca()

	node_weights = som.get_weights()
	rows, cols = node_weights.shape[:2]
	feat_x, feat_y = 0, 1

	for i in range(rows):
		for j in range(cols):
			current = node_weights[i, j]
			# vertical neighbor
			if i + 1 < rows:
				neighbor = node_weights[i+1, j]
				ax.plot([current[feat_x], neighbor[feat_x]],
						[current[feat_y], neighbor[feat_y]], 'k-', alpha=0.3)
			# horizontal neighbor
			if j + 1 < cols:
				neighbor = node_weights[i, j+1]
				ax.plot([current[feat_x], neighbor[feat_x]],
						[current[feat_y], neighbor[feat_y]], 'k-', alpha=0.3)
