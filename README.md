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
1) Basic Linear shrinkage
2) Eigenvalues clipping
3) Rotationally invariant, optimal shrinkage


.Implement the 3 remaining estimator

.What is the theoretical basis on which those estimators have been built?

- Case 2: The correlation matrix is supposed to be time-dependent

