
# K Nearest Neighbor

![](img/wilson.jpg)

## A movie-ing target

## Friend inventory

On a piece of paper, draw a two column table.
- 1st column: names of 9 friends
- 2nd column: whether each friend would recommend Parasite or A Star is Born




![movies](img/movies.png)

![bullseye3](img/bullseye2.png)

## Friend Inventory
### Decision majority by ring

If you just polled the inner ring of people, what movie would you end up seeing?
- How about if you polled the first *and* second ring?
- The first three rings?


## Friend Inventory
### What did you end up seeing?

Who's movie choices changed based on how many people you polled?
![movies](img/movies.png)



## Friend Inventory
### What's the "algorithm" we used for this process, in normal words?

### How does this relate to K nearest neighbor(knn)?

![annotate](img/bullseye-annotate.png)

# What do we have so far in our machine learning toolkit?

![](https://media.giphy.com/media/EX0Bia9M5eUoM/giphy.gif)

- So far, we have two models, linear regression and logistic regression.
They are very different with regards to their target variable. <br>
- But, they are similar in their underlying assumption.  Both assume a linear relationship between target and features. 
- During training, the machine learns the "best-fit" $\beta$-parameters to create a predictive model that outputs the result of a linear equation.

$$ \hat y = \hat\beta_0 + \hat\beta_1 x_1 + \hat\beta_2, x_2 +\ldots + \beta_n x_n $$

Such models are called **parametric**. After training, need the $\beta$-parameters are all you need to be able to predict on new data.

>Other parametric models are:<br>
    1. Naive-Bayes<br>
    2. Linear Discriminant Analysis<br>
    3. Simple neural networks<br>
    

    

## K-nearest neighbors (KNN) classification

> K-Nearest Neighbors, or KNN, is our first example of a **non-parametric model**.

> Non-parametric models assume that the data distribution cannot be defined in
terms of such a finite set of parameters.

> KNN is also a **Lazy learning** or **Instance-based (IB)** algorithm.  IB learning methods simply store the training examples and postpone the generalization (building a model) until a new instance must be classified or prediction made.

> This means:
 - the training time for KNN is very small; the time it takes to store the training data in memory.  
 - predictions are memory intensive, and can take a long time.

# Implementing a K-Nearest Neighbors Model

### How does the KNN algorithm work?

### What should the grey point be?

<img src='img/scenario.png' width=500/>

## KNN has the following basic steps:

<img src='img/knn-process.png' width=500/>

### Voting

How to break ties:

1. When doing a binary classification, often use an odd K to avoid ties.
2. Multiple approaches for Multiclass problems:
    - Reduce the K by 1 to see who wins.
    - Weight the votes based on the distance of the neighbors

### Example training data

This example uses a multi-class problem and each color represents a different class. 

### KNN classification map (K=1)

![1NN classification map](img/knn_1neighbor.png)

K=1 corresponds to [voronoi cells](https://en.wikipedia.org/wiki/Voronoi_diagram)

![](img/vernoi_cells.png)

### KNN classification map (K=3)

![5NN classification map](img/knn_3neighbors.png)


# Euclidean Distance

**Euclidean distance** refers to the distance between two points. These points can be in different dimensional space and are represented by different forms of coordinates. In one-dimensional space, the points are just on a straight number line.



### Measuring distance in a 2-d Space

In two-dimensional space, the coordinates are given as points on the x- and y-axes

![alt text](img/euclidean_2d.png)

## A bit more math
### Measuring distance in a 3-d Space

In three-dimensional space, x-, y- and z-axes are used. 

$$\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 +  (z_1-z_2)^2}$$


## A bit more math
### Euclidean Distance Equation
![alt text](img/euclidean-equation.png)

The source of this formula is in the Pythagorean theorem. 

# Let's code out KNN together using Euclidean distance

# Manhattan distance

Manhattan distance is the distance measured if you walked along a city block instead of a straight line. 

> if 𝑥=(𝑎,𝑏) and 𝑦=(𝑐,𝑑),  
> Manhattan distance = |𝑎−𝑐|+|𝑏−𝑑|

![](img/manhattan.png)

## Implementing the KNN Classifier with SKlearn

## Reviewing the Pima dataset

We are going to revisit the pima native Americans diabetes dataset. Can we use knn to classify people correctly and then predict if someone will have diabetes?
import pandas as pd

## Loading the data


### Importance of Scaling

Scaling is essential for algorithms which depend on distance calculations.

Consider, in the diabetes dataset, how the euclidean distance would change with and without scaling. 

How would the influence of BMI in the distance metric compare to the influence of pregnancies?


#### Should we use a Standard Scaler or Min-Max Scaler?

https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
http://datareality.blogspot.com/2016/11/scaling-normalizing-standardizing-which.html

## scikit-learn 4-step modeling pattern

![steps](img/sklearnsteps.png)

**Step 1:** Import the class you plan to use

**Step 2:** "Instantiate" the "estimator"

- "Estimator" is scikit-learn's term for model
- "Instantiate" means "make an instance of"

**Class specifications**
- Name of the object does not matter
- Can specify tuning parameters (aka "hyperparameters") during this step
- All parameters not specified are set to their defaults

**Step 3:** Fit the model with data (aka "model training")

- Model is learning the relationship between X and y
- Occurs in-place

**Step 4:** Predict the response for a new observation

- New observations are called "out-of-sample" data
- Uses the information it learned during the model training process

## Using a different value for K

#### Search for an optimal value of K for KNN


#### Visual comparison of different $K$s


### What value of K performs best on our Test data?

Here we use F score, what other metrics could we use?

# Grid Search

### How do you think K size relates to our concepts of bias and variance?

![alt text](img/K-NN_Neighborhood_Size_print.png)

# KNN as regression  

KNN can also be used to predict a continuous target variable.
It simply finds the K nearest neighbors for a new point, then predicts the new point as the average of the target values for those k-nearest neighbors.

## Resources

- [Nearest Neighbors](http://scikit-learn.org/stable/modules/neighbors.html) (user guide), [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) (class documentation)

- [Videos from An Introduction to Statistical Learning](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/)
    - Classification Problems and K-Nearest Neighbors (Chapter 2)
    - Introduction to Classification (Chapter 4)
    - Logistic Regression and Maximum Likelihood (Chapter 4)
