# Exercise 1

A solution for the exercise of [01_dataset](./01_dataset.md) by [@KorchmarosDaniel](https://github.com/KorchmarosDaniel).

## Q1

*Why is it important to drop values only after we've selected the features of interest, and not beforehand? What would happen in the other scenario?*

It is important to drop values after selecting the features of interest, as the other way around we may accidentally delete rows that only had data we were never interested anyway. This results in less data overall to work with which can reduce the effectiveness of our model.

### Example 1

| **A** | **B** | **C** |
| ----- | ----- | ----- |
| 1     | 3     | Na    |
| 2     | Na    | 4     |
| 2     | 3     | Na    |

Say we had this simple data with columns *A*, *B* and *C*, however we are only interested in the *A* and *B* attributes.

If we removed all the rows with empty values we would be left with an empty dataset. However, if we specified that we were only interested in the before mentioned attributes, we would still have rows to work with.

| **A** | **B** | **C** |
| ----- | ----- | ----- |
| 1     | 3     | Na    |
| 2     | 3     | Na    |

## Q2

*Why do we need to be careful with this form when calculating similarities between data points (records)? Is there any other commonly used approach to encode categorical variables that bypasses the issue, what is its drawback?*

Assigning each value a unique integer is generally not a good idea, as it can introduce a relationship between data that does not exist. In essence, we cannot explicitly say that one category is "closer to another" or "further away", which is implied if you assign a number to each value, since they are an ordered set. Clustering and other algorithms that use distance assume that the difference between numbers is *meaningfull*

### Example 2

Say we assign the following values for each island:

| **Biscoe** | **Dream** | **Torgersen** |
| ---------- | --------- | ------------- |
| 1          | 2         | 3             |

This is a totally arbitrary assignment, as there is no relationship between the islands and the numbers. Biscoe is not "closer" to Dream, than to Torgersen". They are not "one unit" apart.

### One-Hot Encoding

An alternative approach to handling categorical data is one-hot encoding. In this method we create a new binary column for each value in the category. The above mapping would be replaced with binary rows like this:

| **island_Biscoe** | **island_Dream** | **island_Torgersen** |
| ----------------- | ---------------- | -------------------- |
| 1                 | 0                | 0                    |
| 0                 | 1                | 0                    |

This fixes the problems stated before, however it can drastically increase the dimensions of our dataset, which can slow down algorithms, and most of these rows will be empty.

## Q3

*Why is scaling necessary in general? And especially when working with methods that rely on distance measurements such as K-means or K-Nearest Neighbors?*

Different features often have different units and scales. Without scaling, a feature with large numeric values will dominate the others simply because of its scale, not because it is more important. Scaling ensures that features are on comparable scales, which is what makes the data usable with distance based algorithms.

### Example 3

Say we are looking at penguins and are using their bill length and body mass as features.

- Bill length between 32 and 60
- Body mass between 2700 and 6300

Body mass is around 100 times larger than bill length, meaning without scaling it will make the bill length obsolete.
