NIS_CACHE = .cache/nisapi
TOKEN_PATH = scripts/socrata_app_token.txt
TOKEN = $(shell cat $(TOKEN_PATH))
CONFIG = scripts/config_flu.yaml
RAW_DATA = output/data/nis_raw_flu.parquet
MODEL_FITS = output/fits/model_fits.pkl
DIAGNOSTICS = output/diagnostics/tables/
DIAGNOSTIC_PLOTS = output/diagnostics/plots/
FORECASTS = output/forecasts/tables/forecasts.parquet
FORECAST_PLOTS = output/forecasts/plots/
SCORES = output/scores/tables/scores.parquet
SCORE_PLOTS = output/scores/plots/scores.png

.PHONY: cache

all: $(RAW_DATA) $(MODEL_FITS) $(DIAGNOSTICS) $(DIAGNOSTIC_PLOTS) $(FORECASTS) $(SCORES) $(FORECAST_PLOTS) $(SCORE_PLOTS)

$(FORECAST_PLOTS): scripts/postprocess.py $(FORECASTS) $(RAW_DATA)
	python $< \
		--pred=$(FORECASTS) --obs=$(RAW_DATA) --config=$(CONFIG) \
		--eval=$(SCORES) --output=$(FORECAST_PLOTS) --scores=$(SCORE_PLOTS)

$(SCORES): scripts/eval.py $(FORECASTS) $(RAW_DATA)
	python $< \
		--pred=$(FORECASTS) --obs=$(RAW_DATA) --config=$(CONFIG) \
		--output=$@

$(FORECASTS): scripts/forecast.py $(RAW_DATA) $(MODEL_FITS) $(CONFIG)
	python $< --input=$(RAW_DATA) --models=$(MODEL_FITS) --config=$(CONFIG) --output=$@

$(DIAGNOSTICS): scripts/diagnostics.py $(MODEL_FITS) $(CONFIG)
	python $< --input=$(MODEL_FITS) --config=$(CONFIG) --output_table=$(DIAGNOSTICS) \
	--output_plot=$(DIAGNOSTIC_PLOTS)

$(MODEL_FITS): scripts/fit.py $(RAW_DATA) $(CONFIG)
	python $< --input=$(RAW_DATA) --config=$(CONFIG) --output=$@

$(RAW_DATA): scripts/preprocess.py $(NIS_CACHE) $(CONFIG)
	python $< --cache=$(NIS_CACHE)/clean --config=$(CONFIG) --output=$@

cache: $(NIS_CACHE)/status.txt

$(NIS_CACHE)/status.txt $(TOKEN_PATH):
	python -c "import nisapi; nisapi.cache_all_datasets('$(NIS_CACHE)', '$(TOKEN)')"
	find $(NIS_CACHE)/clean -type f | xargs sha1sum > $@
