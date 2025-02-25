import polars as pl


def standardize(x, mn=None, sd=None):
    """
    Standardize: subtract mean and divide by standard deviation.

    Parameters
    x: pl.Expr | float64
        the numbers to standardize
    mn: float64
        the term to subtract, if not the mean of x
    sd: float64
        the term to divide by, if not the standard deviation of x

    Returns
    pl.Expr | float
        the standardized numbers

    Details
    If the standard deviation is 0, all standardized values are 0.0.
    """
    if type(x) is pl.Expr:
        if mn is not None:
            return (x - mn) / sd
        else:
            return (
                pl.when(x.drop_nulls().n_unique() == 1)
                .then(0.0)
                .otherwise((x - x.mean()) / x.std())
            )
    else:
        if mn is not None:
            return (x - mn) / sd
        else:
            return (x - x.mean()) / x.std()


def unstandardize(x, mn, sd):
    """
    Unstandardize: add standard deviation and multiply by mean.

    Parameters
    x: pl.Expr
        the numbers to unstandardize
    mn: float64
        the term to add
    sd: float64
        the term to multiply by

    Returns
    pl.Expr
        the unstandardized numbers
    """
    return x * sd + mn
