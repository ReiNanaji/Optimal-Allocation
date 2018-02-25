# Optimal-Allocation

Cleaning correlation matrix - Joel Bun, Jean-Phillipe Bouchaud, Marc Potters

"It is well known that large losses at a portfolio level are mostly due to correlated moves of its constituents" - Theory of financial risk and derivative pricing: from statistical physics to risk management.

With the sample estimate: "Small eigenvalues are too small, Large eigenvalues are too large" - Distribution of eigenvalues for some sets of random matrices

What is the best allocation of money? 
- equally weighted 
- mean-variance optimization: the mean is a proxy for the strategy ability to generate return while the variance will be the risk measure. 

The question is how can a find a robust estimator of the out-of-sample risk and the out-of-sample performance?

- Case 1: The correlation matrix is supposed to be time-independent

The matrices are built on in-sample data (obviously)

0) Sample estimation 
The intuition behind the usage of the sample covariance matrix as a estimator for the out-of-sample covariance matrix: 
Let the data speaks! 
If we have enough data, given the law of large numbers, the sample covariance matrix gives an unbiased estimator.

1) Basic Linear shrinkage
The drawback of the previous method lies in the potential lack of data or if the number of stocks is too large. 
In that case, the estimator is affected with estimation error. 
On the other hand, if we model the stock return as a linear model with a known residue distribution, then the covariance matrix can be estimated with less error. HOWEVER, the covariance matrix obtained won't be unbiased as the factors chosen to explain the stock return may not be sufficient/relevant. 
Example: Sharpe model, stock return = linear function of the market return with uncorrelated residue

The idea is that the optimal estimator is a weighted average between the unbiased sample matrix and the biased factor matrix. 
The question being: How is the optimal weight alpha chosen? 

Improved Estimation of the Covariance Matrix of Stock Returns With an Application to Portfolio Selection, Olivier Ledoit

alpha = argmin || alpha * Sharpe covariance + ( 1 - alpha ) * Sample covariance - Real covariance ||

Expected properties of alpha: 
Alpha should be a function of 1/T. 
In fact, if T is large alpha must tend to 0 such that the best estimator is the sample covariance matrix.

2) Eigenvalues clipping
3) Rotationally invariant, optimal shrinkage


.Implement the 3 remaining estimator

.What is the theoretical basis on which those estimators have been built?

- Case 2: The correlation matrix is supposed to be time-dependent

