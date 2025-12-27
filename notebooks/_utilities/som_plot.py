import matplotlib.pyplot as plt
import plotly.graph_objects as go

from typing import Literal
from mpl_toolkits.axes_grid1 import make_axes_locatable

point_style_attributes = {
    "edgecolor": "royalblue",
    "color": "cornflowerblue",
    "linewidth": 1,
    "s": 10,
    "alpha": .75,
}
node_style_attributes = {
    "edgecolor": "red",
    "color": "red",
    "marker": "s",
    "s": 12,
    "alpha": .9,
	"zorder": 10,
}

def visualize_hitmap(som, X, cmap="binary", title="Hitmap", ax=None):
	if ax is None:
		ax = plt.gca()
	
	ax.set_title(title)

	hitmap = som.activation_response(X).astype(int)

	im = ax.imshow(hitmap, origin="lower", cmap=cmap)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="6%", pad=0.2)
	cax.tick_params(labelsize=11)
	cbar = plt.colorbar(im, cax=cax)

def visualize_distance_map(som, X, neighbor_distance_scaling: Literal["sum", "mean"]="mean", cmap="Spectral_r", title="Distance map", ax=None):
	if ax is None:
		ax = plt.gca()
	
	ax.set_title(title)

	distmap = som.distance_map(scaling=neighbor_distance_scaling)

	im = ax.imshow(distmap, origin="lower", cmap=cmap)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="6%", pad=0.2)
	cax.tick_params(labelsize=11)
	cbar = plt.colorbar(im, cax=cax)

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


def plot_3_feature(X_tr, node_weights_flat, X_other=None, other_style_attributes={}, other_label="other", axis_label="Feature", title=None, figsize=(10, 6)):
	fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

	ax = axes[0]
	ax.scatter(X_tr[:, 0], X_tr[:, 1], **point_style_attributes, label="train data (normal)")
	ax.scatter(node_weights_flat[:, 0], node_weights_flat[:, 1], **node_style_attributes, label="SOM node")

	if X_other is not None:
		ax.scatter(X_other[:, 0], X_other[:, 1], **{**point_style_attributes, **other_style_attributes}, label=other_label)
	
	ax.set_xlabel(f"{axis_label} 1")
	ax.set_ylabel(f"{axis_label} 2")

	ax = axes[1]
	ax.scatter(X_tr[:, 0], X_tr[:, 2], **point_style_attributes, label="train data (normal)")
	ax.scatter(node_weights_flat[:, 0], node_weights_flat[:, 2], **node_style_attributes, label="SOM node")

	if X_other is not None:
		ax.scatter(X_other[:, 0], X_other[:, 2], **{**point_style_attributes, **other_style_attributes}, label=other_label)
	
	ax.set_xlabel(f"{axis_label} 1")
	ax.set_ylabel(f"{axis_label} 3")

	ax = axes[2]
	ax.scatter(X_tr[:, 1], X_tr[:, 2], **point_style_attributes, label="train data (normal)")
	ax.scatter(node_weights_flat[:, 1], node_weights_flat[:, 2], **node_style_attributes, label="SOM node")

	if X_other is not None:
		ax.scatter(X_other[:, 1], X_other[:, 2], **{**point_style_attributes, **other_style_attributes}, label=other_label)
	
	ax.set_xlabel(f"{axis_label} 2")
	ax.set_ylabel(f"{axis_label} 3")

	for ax in axes:
		ax.set_aspect("equal")

	if title is not None:
		plt.suptitle(title)

	plt.tight_layout()
	plt.show()


def plot_in_3PC(X_tr, nw_shape, nw_flat):
	fig = go.Figure()

	fig.add_trace(go.Scatter3d(
		x=X_tr[:, 0], y=X_tr[:, 1], z=X_tr[:, 2],
		mode='markers',
		marker=dict(size=3, symbol="square", opacity=.5),
		name='Training data (normal)'
	))

	fig.add_trace(go.Scatter3d(
		x=nw_flat[:, 0], y=nw_flat[:, 1], z=nw_flat[:, 2],
		mode='markers',
		marker=dict(size=5, color='black', opacity=0.1),
		name='SOM nodes'
	))

	fig.update_layout(scene=dict(
		xaxis_title="PC 1",
		yaxis_title="PC 2",
		zaxis_title="PC 3",
	))

	rows, cols, _ = nw_shape
	edges = []

	for i in range(rows):
		for j in range(cols - 1):
			a = i * cols + j
			b = i * cols + (j + 1)
			edges.append((a, b))

	for i in range(rows - 1):
		for j in range(cols):
			a = i * cols + j
			b = (i + 1) * cols + j
			edges.append((a, b))

	for a, b in edges:
		fig.add_trace(go.Scatter3d(
			x=[nw_flat[a, 0], nw_flat[b, 0]],
			y=[nw_flat[a, 1], nw_flat[b, 1]],
			z=[nw_flat[a, 2], nw_flat[b, 2]],
			mode='lines',
			line=dict(color='black', width=2),
			showlegend=False
		))

	fig.update_layout(
		width=900, 
		height=700,
		margin=dict(l=0, r=0, b=0, t=0)
	)

	fig.update_layout(legend=dict(
		yanchor="top",
		y=0.99,
		xanchor="left",
		x=0.01
	))

	camera = dict(
		eye=dict(x=1, y=1.5, z=1.5)
	)
	fig.update_layout(scene_camera=camera)
	fig.show()
