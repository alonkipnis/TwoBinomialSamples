# TwoBinomialSamples: Two-sample testing for counts data
Usually in the context of a multiple testing approach to compare two or more
frequency tables. 

References:
[1] D. L. Donoho and A. Kipnis. (2022) Higher criticism to compare two large frequency tables,
with sensitivity to possible rare and weak differences. Annals of Statistics. 
[2]  C. B. Dean. (1992) Testing for Overdispersion in Poisson and Binomial Regression Models. 
Journal of the American Statistical Association


## Methods:
- ``bin_allocation_test`` (the test from [1])
- ``bin_variance_test`` (test from [2])
- ``bin_variance_test_df`` the same as ``bin_variance_test`` plus additional information

## Example:
```
from scipy.stats import poisson



```
