# Clustering and Visualizing Data in 2D

> [!WARNING]
> To be updated!

A project explored in [Data Science Club at ELTE](https://datasciencelte.netlify.app), 2025.

Coordinated by Matthew Balogh (✉️ matebalogh@student.elte.hu, 🐙 [matthew-balogh](https://github.com/matthew-balogh))

*This project ranges from getting introduced to essential data science concepts in a beginner-friendly environment through more advanced learning (but not Deep Learning) and dimensionality reduction techniques to the contribution of the development of a new Python library.*

## Background

*Machine Learning* has two major branches: *supervised* and *unsupervised learning*. In supervised learning, a decision boundary is learned driven by the available labeled data. In the **unsupervised** setting, we don't have such labels, we only have the characteristics of the records. Instead of learning from the association between the descriptor variables and labels, we group the records by their given traits. The objective is to create groups with similar elements, while being different from elements in other groups.  While supervised learning seems to be more concrete, **unsupervised learning has an advantage**, by being able to detect new patterns.

> [!Tip] Real-life example:
**Supervised:** We may have a collection of spam and not spam e-mails and a model is trained to recognize traits (e.g.: word appearances) in those with spam label. \
**Unsupervised:** Group similar e-mails together without explicitly feeding the model with labels. For example, group e-mails sent by similar companies at the similar time, including similar content.
 
> [!Important] Advantage of unsupervised learning:
If you were to receive newsletter e-mails that you had subscribed to, you may want these e-mails automatically recognized and flagged as "Promotions" as they appear, a task supervised spam/not spam detection could not achieve.


## Topics covered

1. Clustering **two-dimensional** data
2. Clustering **high-dimensional data**, while viusalizing it in **2D**
3. *Contribution to a new visualization method (Optional)*

## Outline of project

1. We will explore **K-means**, a simple clustering algorithm and will apply in simple scenarios such as a 2D coordinate system or small datasets such as the *Iris* or the *Mall Customer* dataset.
2. We will get introduced to the so-called Self-Organizing Maps (SOM), which is a **"Restricted K-means"**. We will visit larger datasets to see the advantage of SOM over a simple K-means.
3. Participants who show dedication to the project and are interested in the visualization of clustering results of high-dimensional data will have the opportunity to contribute to a new visualization method.

## What will you learn from the project?

- introduced to *Unsupervised Learning*
- introduced to *Clustering*
- hands-on experience with *K-means* - a simple and basic clustering algorithm in *Data Science*
- a brief overview of the entire *Data Science pipeline*
- in a beginner-friendly 2D coordinate system


> [!NOTE]
> asd

- introduced to an unpopular yet very powerful extension of *K-means*
- hands-on experience with visualizing high-dimensional data
- hands-on experience with scenarios involving larger datasets such as clustering documents


> [!IMPORTANT] If you are dedicated, you will:

- find limitations of the above mentioned visualization techniques (an objective that we *Data Scientist* and *Researchers* generally have in front of us)
- collaborate and have an influence in developing a new visualization technique in the form of a new Python library
