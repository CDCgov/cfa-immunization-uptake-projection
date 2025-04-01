NIS_CACHE = .cache/nisapi
TOKEN_PATH = scripts/socrata_app_token.txt
TOKEN = $(shell cat $(TOKEN_PATH))
CONFIG = scripts/config.yaml
RAW_DATA = data/nis_raw.parquet
MODEL_FITS = output/model_fits.pkl
DIAGNOSTICS = output/
FORECASTS = data/forecasts.parquet
POSTERIORS = data/posteriors.parquet
SCORES = data/scores.parquet
PROJ_PLOTS = output/projections.png
SUMMARY_PLOTS = output/summary.png
SCORE_PLOTS = output/scores.png

.PHONY: cache

all: $(FORECASTS)

$(PROJ_PLOTS) $(SUMMARY_PLOTS): scripts/postprocess.py $(FORECASTS) $(RAW_DATA)
	python $< \
		--pred=$(FORECASTS) --obs=$(RAW_DATA) --config=$(CONFIG) \
		--proj_output=$(PROJ_PLOTS) --summary_output=$(SUMMARY_PLOTS)

$(FORECASTS): scripts/forecast.py $(RAW_DATA) $(MODEL_FITS) $(CONFIG)
	python $< --input=$(RAW_DATA) --models=$(MODEL_FITS) --config=$(CONFIG) --output=$@

$(DIAGNOSTICS): scripts/diagnostics.py $(MODEL_FITS) $(CONFIG)
	python $< --input=$(MODEL_FITS) --config=$(CONFIG) --output_dir=$@

$(MODEL_FITS): scripts/fit.py $(RAW_DATA) $(CONFIG)
	python $< --input=$(RAW_DATA) --config=$(CONFIG) --output=$@

$(RAW_DATA): scripts/preprocess.py $(NIS_CACHE) $(CONFIG)
	python $< --cache=$(NIS_CACHE)/clean --config=$(CONFIG) --output=$@

cache: $(NIS_CACHE)/status.txt

$(NIS_CACHE)/status.txt $(TOKEN_PATH):
	python -c "import nisapi; nisapi.cache_all_datasets('$(NIS_CACHE)', '$(TOKEN)')"
	find $(NIS_CACHE)/clean -type f | xargs sha1sum > $@
