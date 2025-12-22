import numpy as np

from typing import Literal
from minisom import MiniSom

def train_som(X, d1, d2, sigma, learning_rate, num_iteration, topology="rectangular", use_epochs=True, initial_weights=None, random_order=True, verb=False, random_seed=None):
	som = MiniSom(d1, d2, input_len=X.shape[1], sigma=sigma, learning_rate=learning_rate, topology=topology, random_seed=random_seed)

	if initial_weights is None:
		som.random_weights_init(X)
	else:
		som._weights = initial_weights

	som.train(X, num_iteration=num_iteration, random_order=random_order, use_epochs=use_epochs, verbose=verb)

	if verb:
		QE, TE, TE_VN = calc_som_main_qualities(som, X)
		print("\nBrief quality of SOM:")
		print(f"Quantization error:\t{QE}")
		print(f"Topographic error:\t{TE}")
		print(f"Topographic error (VN):\t{TE_VN}")
	return som

def calc_recommended_grid_size(X):
	N = len(X)
	dd = 5 * np.sqrt(N)
	d = int(np.sqrt(dd).round())
	return dd, d

def calc_som_main_qualities(som:MiniSom, X, digits=3):
	QE = np.round(som.quantization_error(X), digits)
	TE = np.round(som.topographic_error(X), digits)
	ATE = np.round(calc_topographic_error(som, X, neighborhood="von-neumann"), digits)
	return QE, TE, ATE

def calc_topographic_error(som:MiniSom, data, neighborhood:Literal["moore", "von-neumann"]="moore", aggregated=True):
	total_neurons = np.prod(som._activation_map.shape)
	if total_neurons == 1:
		print('The topographic error is not defined for a 1-by-1 map.')
		return np.nan
	if som.topology == 'hexagonal':
		return np.nan
	else:
		return topographic_error_rectangular(som, data, neighborhood, aggregated)
	
def topographic_error_rectangular(som:MiniSom, X, neighborhood:Literal["moore", "von-neumann"]="moore", aggregated=True):
	t = 1.42 if neighborhood == "moore" else 1
	# b2mu: best 2 matching units
	b2mu_inds = np.argsort(som._distance_from_weights(X), axis=1)[:, :2]
	b2my_xy = np.unravel_index(b2mu_inds, som._weights.shape[:2])
	b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1]
	dxdy = np.hstack([np.diff(b2mu_x), np.diff(b2mu_y)])
	distance = np.linalg.norm(dxdy, axis=1)
	return (distance > t).mean() if aggregated is True else (distance > t)
