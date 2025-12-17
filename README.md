# Clustering Data and Visualizing in 2D

A project explored in [Data Science Club at ELTE](https://datasciencelte.netlify.app), 2025.

Coordinated by *Matthew Balogh* (✉️ matebalogh@student.elte.hu, 🐙 [matthew-balogh](https://github.com/matthew-balogh))

**Brief objective:** Group similar data points together and then visualize *where* and *how* those groups separate from each other and explore what characterizes them.

## Topics covered

1. Clustering **two-dimensional** data with *K-means*
2. Clustering **high-dimensional data**, while visualizing it in **2D** using *Self-Organizing Maps*
3. Touch up on how *Self-Organizing Maps* relate to neural networks

Our analysis will be accompanied by visualizations whether it is related to simple or high-dimensional scenarios.

Also, by visiting the above mentioned steps, we expect that by the end of the project we will have both theoretical knowledge and practical experience in the field of *Representation Learning*.

## Documentation

Background information and experimental details are recorded in [Documentation.md](./Documentation.md).

## Team member responsibilities

### Theoretical part

| Assignee               | Topic                                                                                                                                         | Status |
| :--------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
| **Hellman, Barnabas**  | _"General Data Science Pipeline"_ and _"Unsupervised Learning"_                                                                               |   ✅    |
| **Korchmaros, Daniel** | _"Representation / Prototype Learning"_                                                                                                       |   ✅    |
| **Hodali, Bishara**    | _"Neural networks in general and in Clustering"_ and _"Why Self-Organizing Maps (SOM) can be considered as a neural network (and why not?!)"_ |   ✅    |
| **Balogh, Mate**       | _"K-means and Self-Organizing Maps"_                                                                                                          |   ⏳    |

### Practical part

The first round of exercises are listed in [applied-exercises/01_dataset.md](./applied-exercises/01_dataset.md).

The related dataset and its brief exploration can be found in the [penguins.ipynb](./datasets/penguins/simple/penguins.ipynb) notebook.

| Topic                          | Solved |
| :----------------------------- | :----: |
| Dropping missing values        |   ✅    |
| Encoding categorical variables |   ✅    |
| Scaling the data               |   ✅    |
| Dimensionality reduction       |   -    |

As for the main practical tasks:

| Task                                               | Status |
| :------------------------------------------------- | :----: |
| Clustering with _K-Means_                          |   ✅   |
| Evaluate _K-Means_ clustering against true labels  |   ✅   |
| Representation learning with _SOM_                 |   ⏳   |
| Clustering via _SOM_ representation                |   ⏳   |
| Evaluate _SOM_ clustering against true labels      |   ⏳   |
| Compare _K-Means_ and _SOM_ performance            |   ⏳   |
| Enhance the visualization of _SOM_ map             |   ⏳   |
| Investigate metrics that could drive a better representation / clustering |   ?   |

## For contributors

```bash
# 1. install required libraries
pip install -r requirements.txt

# 2. Run a selected `.ipynb` notebook from the project
```
