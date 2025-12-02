## Supervised vs. Unsupervised Learning
---

<!-- Models usually fall into two categories: *Supervised* and *Unsupervised*. Supervised models have a known outcome, it has labeled data. For example, a model trained on spam emails would include examples of both spam and non-spam emails, from which it „learns” what a spam email looks like. Unsupervised models, on the other hand, don’t have such labels, they compare patterns and trends in data. For example, if you wanted to know the shopping habits of your customers based on their preferences. Unsupervised learning can be challenging as its output can be vague and inconcrete which can lead to bias. It's evaluation is completely different than that of supervised approaches. Still unsupervised learning has the advantage to spot hiddenor previously not seen paterns, that supervised models cannot recognize unless specifically trained to do so. -->

In supervised feature learning, the input data is labelled, meaning data given to the model includes input-label pairs. This resorts in a representation with high label prediction accuracy. Examples include supervised neural networks, multilayer perceptrons, and dictionary learning.

Unsupervised feature learning uses unlabelled input data, it learns features by analyzing the relationship between data points. Examples include Clustering (K-means), ICA and matrix factorization.

In self-supervised learning features are learned using unlabelled data like unsupervised learning, however input-label pairs are constructed from each data point, enabling learning the structure of the data through supervised methods. Examples include word embeddings and autoencoders.
