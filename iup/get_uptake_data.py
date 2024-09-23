import polars as pl
import datetime as dt


def get_uptake_data(
    file_name,
    state_col,
    date_col,
    cumulative_col,
    state_key,
    date_format,
    start_date,
    filters=None,
):
    """
    Imports and formats past cumulative uptake data from a file or url.

    Parameters
    ----------
    file_name : str
        Path or url address to the .csv file of input data
    state_col : str
        Name of the .csv column containing the geographic region
    date_col : str
        Name(s) of the .csv column(s) containing the report date
    cumulative_col : str
        Name of the .csv column giving cumulative uptake
    state_key : str
        Path to .csv file with columns "Abbr" and "Full"
        for geographic region names
    date_format : str
        Format of the dates in date_col, e.g. "%m-%d-%Y",
        ignored if dates are split into multiple columns
    start_date : str
        Date of the season's immunization rollout, in date_format
    filters : Optional[dict]
        Dictionary of filters to apply to the data, where
        keys are column names and values are (lists of) acceptable entries

    Details
    -------

    The data is imported and optionally filtered. Only geographic regions
    named in the state key are kept, and their names are abbreviated.
    Each report data is coerced to "%Y-%m-%d" format. If multiple columns
    are used to give date ranges, the final date in the range is used.
    Season the calendar year during the autumn portion of the disease season.
    Number of days elapsed since rollout, time intervals between reports,
    incident uptake on each report, per-day incident uptake, and the previous
    report's per-day incident uptake are also recorded as columns.

    Returns
    -------
    DataFrame
        Past cumulative uptake information.
    """
    data = (
        (pl.read_csv(file_name))
        .with_columns(cumulative=pl.col(cumulative_col).cast(pl.Float64, strict=False))
        .drop_nulls(subset=["cumulative"])
    )

    if filters is not None:
        filter_expr = pl.lit(True)
        for k, v in filters.items():
            filter_expr &= pl.col(k).is_in(pl.lit(v))
        data = data.filter(filter_expr)

    state_key = pl.read_csv(state_key)
    data = data.with_columns(
        state=pl.col(state_col).replace(state_key["Full"], state_key["Abbr"])
    ).filter(pl.col("state").is_in(state_key["Abbr"]))

    if type(date_col) is list:
        data = data.with_columns(
            pl.col(date_col).cast(pl.String),
            date=pl.concat_str(pl.col(date_col), separator=" "),
        ).with_columns(
            pl.col("date")
            .str.split("-")
            .list.get(1)
            .str.strip_chars()
            .str.to_date(format="%B %e %Y")
        )
    else:
        data = data.with_columns(date=pl.col(date_col).str.to_date(format=date_format))

    start_date = dt.datetime.strptime(start_date, date_format)
    data = data.with_columns(
        elapsed=(pl.col("date") - start_date).cast(pl.Float64)
        / (1000000 * 60 * 60 * 24),
        season=pl.col("date").dt.year()
        + pl.when(pl.col("date").dt.month() < 7).then(-1).otherwise(0),
    )

    data = data.select(["state", "date", "elapsed", "season", "cumulative"]).sort(
        "state", "date"
    )

    data = (
        data.with_columns(
            incident=(pl.col("cumulative") - pl.col("cumulative").shift(1))
            .fill_null(pl.col("cumulative").first())
            .over("state"),
            interval=pl.col("elapsed").diff().fill_null(pl.col("elapsed").first()),
        )
        .with_columns(daily=(pl.col("incident") / pl.col("interval")))
        .with_columns(previous=pl.col("daily").shift(1))
    )

    return data
