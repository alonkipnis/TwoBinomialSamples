import numpy as np
from scipy.stats import binom, norm, poisson
from scipy.stats import binom_test as btest
import pandas as pd

"""
def binom_test_two_sided(x, n, p):
    :x:   number of observed successes
    :n:   number of trails
    :p:   probability of success

    Returns:
    --------
    Prob( |Bin(n,p) - np| >= |x-np| )

    Note: for small values of Prob there are differences
    fron scipy.python.binom_test. It is unclear which one is
    more accurate.
    
    n = n.astype(int)

    x_low = n * p - np.abs(x - n * p)
    x_high = n * p + np.abs(x - n * p)

    p_up = binom.cdf(x_low, n, p) + binom.sf(x_high - 1, n, p)

    prob = np.minimum(p_up, 1)
    return prob * (n != 0) + 1. * (n == 0)
"""


def poisson_test(x, lmd, randomize=False):
    """
    Prob( Pois(lmd) >= x ) + randomization
    """

    p_up = 1 - poisson.cdf(x, lmd) + poisson.pmf(x, lmd)

    x = x.astype(int)

    if randomize:
        p_down = 1 - poisson.cdf(x, lmd)
        U = np.random.rand(x.shape[0])
        prob = (x != 0) * np.minimum(p_down + (p_up - p_down) * U, 1) + U * (x == 0)
    else:
        prob = p_up
    return prob


def binom_test_two_sided(x, n, p, randomize=False):
    """
    :x:   number of observed successes
    :n:   number of trails
    :p:   probability of success
    :randomize:   whether to do a randomized test or not
                  a randomized test returns a P-value that
                  is uniformly distributed if x ~ Bin(n,p)

    Returns:
        :pval:  if randomize=False, pval=Prob( |Bin(n,p) - np| >= |x-np| )
                if randomize=True, pval is a random number such that
            Prob(
            |Bin(n,p) - np| >= |InBinCDF(pval|n, p) - n p|
            ) ~ U(0,1)

    Note: The expression pval=Prob( |Bin(n,p) - np| >= |x-np| ) is not accurate
        when p is close to either 0 or 1. For such cases, it is better to use
        scipy.python.binom_test which is essentially the R function binom_test

    """

    x_low = n * p - np.abs(x - n * p)
    x_high = n * p + np.abs(x - n * p)

    n = n.astype(int)
    p_up = binom.cdf(x_low, n, p) + binom.sf(x_high - 1, n, p)

    if randomize:
        p_down = binom.cdf(x_low - 1, n, p) + binom.sf(x_high, n, p)
        U = np.random.rand(x.shape[0])  # uniform random variable
        prob = (n != 0) * np.minimum(p_down + (p_up - p_down) * U, 1) + U * (n == 0)
    else:
        prob = np.minimum(p_up, 1)
    return prob


def binom_test(x, n, p, alt='two-sided'):
    """
    :x:   number of observed successes
    :n:   number of trails
    :p:   probability of success

    Returns:
    --------
    Prob(Bin(n,p) >= x) ('greater')
    or Prob(Bin(n,p) <= x) ('less')

    Note: for small values of Prob there are differences
    fron scipy.python.binom_test.
    """
    n = n.astype(int)
    if alt == 'greater':
        return binom.sf(x, n, p) + binom.pmf(x, n, p)
    if alt == 'less':
        return binom.cdf(x, n, p)
    if alt == 'two-sided':
        return binom_test_two_sided(x, n, p)


def binom_test_two_sided_slow(x, n, p):
    """
    :x:   number of observed successes
    :n:   number of trails
    :p:   probability of success

     Uses scipy.stats.binom_test on each entry of
     an array. Slower than binom_test_two_sided but
     possibly more accurate
    """

    def my_func(r):
        return btest(r[0], r[1], r[2])

    a = np.concatenate([np.expand_dims(x, 1),
                        np.expand_dims(n, 1),
                        np.expand_dims(p, 1)],
                       axis=1)

    pv = np.apply_along_axis(my_func, 1, a)

    return pv




def bin_variance_test_df(c1, c2, sym=False, max_m=-1):
    """
    Binomial variance test along stripes.
        This version returns all sub-calculations

    Args:
    ----
    :c1:, :c2:   count data
    :sym:  indicates whether the size of both sample is assumed
          identical, hence p=1/2
    :max_m:  we only consider stripes with total counts smaller than max_m
    """

    df_smp = pd.DataFrame({'n1': c1, 'n2': c2})
    df_smp.loc[:, 'N'] = df_smp.agg('sum', axis='columns')

    if max_m > 0:
        df_smp = df_smp[df_smp.n1 + df_smp.n2 <= max_m]

    df_hist = df_smp.groupby(['n1', 'n2']).count().reset_index()
    df_hist.loc[:, 'm'] = df_hist.n1 + df_hist.n2
    df_hist = df_hist[df_hist.m > 0]

    df_hist.loc[:, 'N1'] = df_hist.n1 * df_hist.N
    df_hist.loc[:, 'N2'] = df_hist.n2 * df_hist.N

    df_hist.loc[:, 'NN1'] = df_hist.N1.sum()
    df_hist.loc[:, 'NN2'] = df_hist.N2.sum()

    df_hist = df_hist.join(df_hist.filter(['m', 'N1', 'N2', 'N']).groupby('m').agg('sum'),
                           on='m', rsuffix='_m')
    if max_m == -1:
        df_hist = df_hist[df_hist.N_m > np.maximum(df_hist.n1, df_hist.n2)]

    if sym:
        df_hist.loc[:, 'p'] = 1 / 2
    else:
        df_hist.loc[:, 'p'] = df_hist['NN1'] / (df_hist['NN1'] + df_hist['NN2'])

    df_hist.loc[:, 's'] = (df_hist.n1 - df_hist.m * df_hist.p) ** 2 * df_hist.N
    df_hist.loc[:, 'Es'] = df_hist.N_m * df_hist.m * df_hist.p * (1 - df_hist.p)
    df_hist.loc[:, 'Vs'] = 2 * df_hist.N_m * df_hist.m * (df_hist.m) * (df_hist.p * (1 - df_hist.p)) ** 2
    df_hist = df_hist.join(df_hist.groupby('m').agg('sum').s, on='m', rsuffix='_m')
    df_hist.loc[:, 'z'] = (df_hist.s_m - df_hist.Es) / np.sqrt(df_hist.Vs)
    # df_hist.loc[:,'pval'] = df_hist.z.apply(lambda z : norm.cdf(-np.abs(z)))
    df_hist.loc[:, 'pval'] = df_hist.z.apply(lambda z: norm.sf(z))

    # handle the case m=1 seperately
    n1 = df_hist[(df_hist.n1 == 1) & (df_hist.n2 == 0)].N.values
    n2 = df_hist[(df_hist.n1 == 0) & (df_hist.n2 == 1)].N.values
    if len(n1) + len(n2) >= 2:
        df_hist.loc[df_hist.m == 1, 'pval'] = binom_test_two_sided(n1, n1 + n2, 1 / 2)[0]

    return df_hist


def bin_variance_test(c1, c2, sym=False, max_m=-1):
    """ Binmial variance test along stripes
    Args:
    ----
    c1, c2 : list of integers represents count data from two sample
    sym : flag indicates wether the size of both sample is assumed
          identical, hence p=1/2
    """
    df_hist = bin_variance_test_df(c1, c2, sym=sym, max_m=max_m)
    return df_hist.groupby('m').pval.mean()


def bin_allocation_test(c1, c2, randomize=False,
                        sym=False, alt='two-sided',
                        ret_p=False):
    """ feature by feature exact binomial test
    Args:
    ----
    c1, c2 : list of integers represents count data from two sample
    randomize : flag indicate wether to use randomized P-values
    sym : flag indicates wether the size of both sample is assumed
          identical, hence p=1/2
    alt :  how to compute P-values.
    """

    T1 = c1.sum()
    T2 = c2.sum()

    den = (T1 + T2 - c1 - c2)
    if den.sum() == 0:
        return c1 * np.nan

    p = ((T1 - c1) / den) * (1 - sym) + sym * 1. / 2

    if alt == 'greater' or alt == 'less':
        pvals = binom_test(c1, c1 + c2, p, alt=alt)
        if randomize:
            raise Warning(f"""randomization for alt={alt} is not implemented.
            Pass `randomize=False` to shut down warning""")
    else:  # alt == 'two-sided'
        pvals = binom_test_two_sided(c1, c1 + c2, p, randomize=randomize)

    if ret_p:
        return pvals, p
    return pvals
