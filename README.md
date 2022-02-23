# HCtest -- Higher Criticism Test

Higher Criticism (HC) test statistics for testing the global significance of many independent hypotheses. As an input, 
the test receives a list of P-values and returns the HC test statistics. See (Donoho & Jin 2004)

## Example:
```
from scipy.stats import norm
from multitest import MultiTest

n = 1000 #number of samples

X = norm.rvs(size=n)
pvals = norm.sf(X)

mt = MultiTest(pvals)
hc_val, p_th = mt.HCstar(gamma = 0.25)
minus_log_min_pval = mt.minp()
fdr, pval_fdr = mt.fdr()

print(f"Higher-Criticism test statistic = {hc_val}")
print(f"False-discovery rate {q} critical P-value = {pval_fdr}")
print(f"Bonferroni {minus_log_min_pval}")
```
