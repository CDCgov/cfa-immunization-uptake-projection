NIS_CACHE = .cache/nisapi
TOKEN_PATH = scripts/socrata_app_token.txt
TOKEN = $(shell cat $(TOKEN_PATH))
CONFIG = scripts/config.yaml
RAW_DATA = data/nis_raw.parquet
FORECASTS = data/forecasts.parquet
SCORES = data/scores.parquet
PROJ_PLOTS = output/projections.png
SCORE_PLOTS = output/scores.png

.PHONY: cache

all: $(PROJ_PLOTS) $(SCORE_PLOTS)

$(PROJ_PLOTS) $(SCORE_PLOTS): scripts/postprocess.py $(FORECASTS) $(RAW_DATA) $(SCORES)
	python $< --pred=$(FORECASTS) --obs=$(RAW_DATA) --score=$(SCORES) --proj_output=$(PROJ_PLOTS)  --score_output=$(SCORE_PLOTS)

# $(SCORE_PLOTS): scripts/score_plot.py $(SCORES)
# 	python $< --score=$(SCORES) --output=$@

# $(PROJ_PLOTS): scripts/proj_plot.py $(FORECASTS)
# 	python $< --pred=$(FORECASTS) --obs=$(RAW_DATA) --output=$@

$(SCORES): scripts/eval.py $(FORECASTS)
	python $< --pred=$(FORECASTS) --obs=$(RAW_DATA) --output=$@

$(FORECASTS): scripts/forecast.py $(RAW_DATA)
	python $< --input=$(RAW_DATA) --output=$@

$(RAW_DATA): scripts/preprocess.py cache
	python $< --cache=$(NIS_CACHE)/clean --output=$@

cache: $(NIS_CACHE)/status.txt

$(NIS_CACHE)/status.txt $(TOKEN_PATH):
	python -c "import nisapi; nisapi.cache_all_datasets('$(NIS_CACHE)', '$(TOKEN)')"
	find $(NIS_CACHE)/clean -type f | xargs sha1sum > $@
