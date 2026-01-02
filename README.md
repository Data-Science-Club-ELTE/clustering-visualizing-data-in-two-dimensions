# Clustering Data and Visualizing in 2D

A project explored in [Data Science Club at ELTE](https://datasciencelte.netlify.app), 2025.

Coordinated by *Matthew Balogh* (✉️ matebalogh@student.elte.hu, 🐙 [@matthew-balogh](https://github.com/matthew-balogh))

In this project, titled as *Clustering Data and Visualizing in 2D*, we aim to group similar data points of a dataset together and then visualize *where* and *how* those groups separate from each other. From a theoretical point of view, we plan to explore concepts such as:

- General Data Science pipeline
- Unsupervised learning
- Representation learning
- Neural networks,

while from the practical aspect, we expect to get hands on experience in using:

- *K-means*
- *Self-Organizing Map (SOM)* algorithms
## Collaborators

|                        | Contribution to theoretic overview                              | Contribution to experiments                                                                |
| :--------------------- | :-------------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| **Hellman, Barnabas**  | _"General Data Science Pipeline"_ and _"Unsupervised Learning"_ | -                                                                                          |
| **Korchmaros, Daniel** | _"Representation / Prototype Learning"_                         | K-means clustering                                                                         |
| **Hodali, Bishara**    | _"Neural networks in general and in Clustering"_                | Representation learning via SOM                                                            |
| **Balogh, Mate**       | _"K-means and Self-Organizing Maps"_                            | Refine SOM representation and clustering via SOM. Evaluation and comparison of clusterings |

## Documentation

The detailed documentation can be found at the [Documentation page](https://data-science-club-elte.github.io/clustering-visualizing-data-in-two-dimensions).

## Results and Findings

### Performance of clustering methods

| Experiment                   | Homogeneity | Completeness | V-measure |  Accuracy |
| :--------------------------- | ----------: | -----------: | --------: | --------: |
| K-means clustering           |       0.798 |        0.773 |     0.786 |     0.915 |
| SOM-based clustering         |       0.896 |        0.906 |     0.901 |     0.974 |
| SOM-based (tuned) clustering |   **0.920** |    **0.936** | **0.928** | **0.980** |

### K-means clustering:

- visualizing high-dimensional clustering in 2D plots can be deceitful due to dimensionality collapse
- regardless, comparing clustering and ground truth labels is still possible
- 2D observations required 6 individual plots for mere 4 dimensions

### Representation learning via Self-Organizing Map

- an organized SOM that fits the data close-enough can be used as a two-dimensional window to the high-dimensional dataset
- *PCA* can be used to verify convergence but relying on the first few components may hide information about patterns in the dataset, including irregularly-shaped clusters
- our first attempt with the SOM visual, trained unconventionally, exhibited an interesting pattern that we referred to as the *Principal Curve* of the data

### Clustering via segmenting the Self-Organizing Map representation

- the SOM representation was recreated with hyperparameters optimized for the downstream segmentation task
- the representation is likely to have more nodes than target clusters, resulting in multiple nodes defining a target cluster, which can be identified in the two-dimensional visual by the human eye
- for a more precise clustering, agglomerative clustering was used to label the nodes and segment the SOM grid into 3 regions that resulted in better clustering performance

### Enhanced visualization

- the obtained two-dimensional SOM visualization was enhanced using the 🪷 [lilypond](https://github.com/matthew-balogh/lilypond) library developed by 🐙 [@matthew-balogh](https://github.com/matthew-balogh)


## For collaborators

### Project structure

- `./datasets`: Dataset preprocessing notebooks and files
- `./docs`: Documentation files
- `./exercises`: Exercise notebooks
- `./notebooks`: Notebooks and resources related to the experiments
- `./theory`: Resources for the background section of the documentation
- `./requirements.txt`: Required libraries to run the notebooks

### Instructions to run the notebooks

```bash
# 1. install required libraries
pip install -r requirements.txt

# 2. Run a selected `.ipynb` notebook from the project
```
