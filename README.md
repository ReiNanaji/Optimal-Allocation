# Optimal-Allocation

Cleaning correlation matrix - Joel Bun, Jean-Phillipe Bouchaud, Marc Potters

"It is well known that large losses at a portfolio level are mostly due to correlated moves of its constituents" - Theory of financial risk and derivative pricing: from statistical physics to risk management.

Markowitz approach: find the portfolio whose mean return is maximized while the variance of the return is minimized.

The Markowitz solution results in over-allocation on low variance modes.

The mean vector and the correlation matrix are estimated using historical data. 
We want a correlation matrix that represents the future. 

The question is: How can we find a robust estimator of the correlation matrix?

With the sample estimate: "Small eigenvalues are too small, Large eigenvalues are too large" - Distribution of eigenvalues for some sets of random matrices

- Case 1: The correlation matrix is supposed to be time-independent

1) Basic Linear shrinkage
2) Advanced Linear shrinkage
3) Eigenvalues clipping
4) Eigenvalues substitution
5) Rotationally invariant, optimal shrinkage

- Case 2: The correlation matrix is supposed to be time-dependent

