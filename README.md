# Immunization uptake projections

This repo contains statistical tools to predict the uptake of immunizations (primarily vaccines and boosters). The three primary steps are:

1. Import data sets on past uptake and cast them into a standardize format
2. Fit a variety of models that both capture past uptake as well as project future uptake, and
3. Evaluate model projections against realized uptake.

All three steps are currently under development.

This approach is applicable to seasonal adult immunizations. Each year, the uptake process starts afresh on the immunization rollout date, and individuals' transitions across age groups are not relevant.

## Data sources

Use <https://github.com/CDCgov/nis-py-api> for access to the NIS data.

## Getting started

1. Set up a virtual environment with `poetry shell`
2. Installed the required dependencies with `poetry install`
3. Get a [Socrata app token](https://github.com/CDCgov/nis-py-api?tab=readme-ov-file#getting-started) and save it in `scripts/socrata_app_token.txt`
4. Cache NIS data with `make cache`
5. Copy the config template in `scripts/config_template.yaml` (e.g., to `scripts/config.yaml`) and fill in the necessary fields
    - data: specify the type of vaccine data in terms of: rollout time frame, grouping factors including geography, demography (domain_type, domain), and vaccine type(indicator_type, indicator).
    - forecast_timeframe: specify the start and the end of forecast dates, and interval between forecast dates (*d).
    - evaluation_timeframe: specify the interval of the forecast dates for evaluation. If blank, no evaluation score will be returned.
    - models: specify the name of the model (refer to iup.models), random seed, initial values of parameters, and parameters to use NUTS kernel in MCMC run
    - score_funs: specify the evaluation metrics. Can be a list including "mspe", "mean_bias" and "eos_abe".
6. `make all` to get cleaned data "data/nis_raw.parquet", forecasts "data/forecasts.parquet", evaluation scores "data/scores.parquet", forecast plot "output/projections.png", and evaluation score plot "output/scores.png".

#### Evaluation scores
- Mean squared prediction error on incident projections ("mspe")
- Mean bias on incident projections ("mean_bias")
- Absolute error of end-of-season uptake on incident projections ("eos_abe")

#### Package workflow:

```mermaid

flowchart TB

nis_data(nis_raw.parquet)
forecast(forecasts.parquet)
scores(scores.parquet)
config{{config.yaml}}
proj_plot[projections.png]
score_plot[scores.png]

subgraph raw data
NIS
end

subgraph clean data with seasons
nis_data
end

subgraph prediction by seasons
forecast
end

subgraph evaluation scores
scores
end

subgraph prediction plots
proj_plot
end

subgraph score plots
score_plot
end

NIS -->preprocess.py --> nis_data
nis_data --> forecast.py --> forecast
forecast --> eval.py --> scores
scores --> postprocess.py --> score_plot
nis_data --> postprocess.py --> proj_plot
forecast --> postprocess.py

config --> preprocess.py
config --> forecast.py
config --> eval.py


style nis_data fill:#7f00ff
style forecast fill:#7f00ff
style scores fill:#7f00ff
style config fill:#f58742
style preprocess.py fill:#0080ff
style forecast.py fill:#0080ff
style eval.py fill:#0080ff
style postprocess.py fill:#0080ff
style proj_plot fill:#ff6666
style score_plot fill: #ff6666


```

## Project admins

- Edward Schrom (CDC/CFA/Predict) <tec0@cdc.gov>

## General Disclaimer

This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm). GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

## Public Domain Standard Notice

This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice

This repository is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice

This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice

Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice

This repository is not a source of government records but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).
