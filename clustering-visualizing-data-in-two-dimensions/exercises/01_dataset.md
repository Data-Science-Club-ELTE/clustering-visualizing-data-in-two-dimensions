# Exercise 1 | Dataset

## Introduction

Please check the preprocessed dataset in the [penguins.ipynb](../datasets/penguins/simple/penguins.ipynb) notebook.

If you want to further explore this data, load the exported versions in a separate notebook as:

```python
import pandas as pd

# original version of unlabeled data
X = pd.read_csv("datasets/penguins/simple/X.csv", index_col=0, header=0)

# scaled version of unlabeled data
X_scaled = pd.read_csv("datasets/penguins/simple/X_scaled.csv", index_col=0, header=0)

# labels
y = pd.read_csv("datasets/penguins/simple/y.csv", index_col=0, header=0)

# ...

# convert to numpy if necessary
X_scaled = X_scaled.to_numpy()
y = y['species'].to_numpy()
```

## Exercise 1

The [penguins (simple)](../datasets/penguins/simple/penguins.ipynb) dataset was checked for missing values, which were discarded as follows:

```python
predictors_simple.dropna(inplace=True)
```

That is, records (rows) having missing value in **any of their features** (columns) was dropped from the `predictors_simple` dataset object.

**Question:** *Why is it important to drop values only after we've selected the features of interest, and not beforehand? What would happen in the other scenario?*

## Exercise 2

The *categorical variables* such as *island*, *sex* were excluded from the simple dataset. If we were to learn from those variables too, we would need to convert them into numerical form. One way is to assign a number to each unique value.

**Question:** *Why do we need to be careful with this form when calculating similarities between data points (records)? Is there any other commonly used approach to encode categorical variables that bypasses the issue, what is its drawback?*

## Exercise 3

The features in the dataset were scaled using standardization as follows:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(predictors_simple)

X_scaled = scaler.transform(predictors_simple)
```

**Question:** *Why is this necessary in general? And especially when working with methods that rely on distance measurements such as K-means or K-Nearest Neighbors?*

## Exercise 4

Even if this dataset is a simple one with only 4 features, we cannot visualize the data in its original form. We can use tricks like colors and different markers to indicate additional dimensions of the dataset, but in general, we cannot go beyond the two- or three-dimensional coordinate system that is interpretable for the human eye.

However, there are *dimensionality reduction* techniques that try to represent the data in smaller dimensions with as little information loss as possible. Also, if a simpler representation of the data is learned that comes with spatial information between representative points, we could visualize this prototype, this skeleton of the dataset like just **Self-Organizing Maps** enable us to do so.

**Question:** *What are the few common dimensionality reduction techniques? What if even the reduced form cannot be visualized in 2D-3D, how does the visual "lie" to us?*
