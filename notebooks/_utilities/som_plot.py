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

def plot_three_variables_pairwise(layers:list[dict]=[],
								  axis_label="Feature", title=None, figsize=(10, 6)):
	fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

	default_style_attributes = {"color": "none", "edgecolor": "cornflowerblue", "s": 30, "alpha": .9}

	for li, l in enumerate(layers):
		X = l.get('data', None)
		style_attributes = l.get('styles', {})
		label = l.get('label', None)

		style_attributes = {**default_style_attributes, **style_attributes}

		ax = axes[0]
		ax.scatter(X[:, 0], X[:, 1], **style_attributes, label=label)
		ax.set_xlabel(f"{axis_label} 1")
		ax.set_ylabel(f"{axis_label} 2")

		ax = axes[1]
		ax.scatter(X[:, 0], X[:, 2], **style_attributes, label=label)
		ax.set_xlabel(f"{axis_label} 1")
		ax.set_ylabel(f"{axis_label} 3")

		ax = axes[2]
		ax.scatter(X[:, 1], X[:, 2], **style_attributes, label=label)
		ax.set_xlabel(f"{axis_label} 2")
		ax.set_ylabel(f"{axis_label} 3")

	for ax in axes:
		ax.set_aspect("equal")

	ax.legend(loc='center left', bbox_to_anchor=(1.02, .93))

	if title is not None:
		plt.suptitle(title)

	plt.tight_layout()
	plt.show()
	

def plot_in_3PC(layers:list[dict]=[]):
	fig = go.Figure()

	fig.update_layout(scene=dict(
		xaxis_title="PC 1",
		yaxis_title="PC 2",
		zaxis_title="PC 3",
	))

	for li, l in enumerate(layers):
		lData = l.get('data', [])
		lMarker = l.get('marker', dict(size=3, opacity=.9, color="cornflowerblue"))
		lLabel = l.get('label', f"Layer {li}")

		fig.add_trace(go.Scatter3d(
			x=lData[:, 0], y=lData[:, 1], z=lData[:, 2],
			mode='markers',
			marker=lMarker,
			name=lLabel
		))

		lNodeWeights = l.get('nodeWeights', None)
		lGridShape = l.get('gridShape', None)

		if lNodeWeights is not None and lGridShape is not None:

			fig.add_trace(go.Scatter3d(
				x=lNodeWeights[:, 0], y=lNodeWeights[:, 1], z=lNodeWeights[:, 2],
				mode='markers',
				marker=dict(size=5, color='black', opacity=0.1),
				name='SOM nodes'
			))

			rows, cols, _ = lGridShape
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
					x=[lNodeWeights[a, 0], lNodeWeights[b, 0]],
					y=[lNodeWeights[a, 1], lNodeWeights[b, 1]],
					z=[lNodeWeights[a, 2], lNodeWeights[b, 2]],
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
