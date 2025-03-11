NIS_CACHE = .cache/nisapi
TOKEN_PATH = scripts/socrata_app_token.txt
TOKEN = $(shell cat $(TOKEN_PATH))
CONFIG = scripts/config.yaml
RAW_DATA = data/nis_raw.parquet
FORECASTS = data/forecasts.parquet
SCORES = data/scores.parquet
PROJ_PLOTS = output/projections.png
SUMMARY_PLOTS = output/summary.png
SCORE_PLOTS = output/scores.png

.PHONY: cache

# all: $(PROJ_PLOTS) $(SCORE_PLOTS)
all: $(PROJ_PLOTS) $(SUMMARY_PLOTS)
# $(PROJ_PLOTS) $(SCORE_PLOTS): scripts/postprocess.py $(FORECASTS) $(RAW_DATA) $(SCORES)
# 	python $< \
# 		--pred=$(FORECASTS) --obs=$(RAW_DATA) --score=$(SCORES) \
# 		--proj_output=$(PROJ_PLOTS) --score_output=$(SCORE_PLOTS)

# $(SCORES): scripts/eval.py $(FORECASTS) $(CONFIG)
# 	python $< --pred=$(FORECASTS) --obs=$(RAW_DATA) --config=$(CONFIG) --output=$@

$(PROJ_PLOTS) $(SUMMARY_PLOTS): scripts/postprocess.py $(FORECASTS) $(RAW_DATA)
	python $< \
		--pred=$(FORECASTS) --obs=$(RAW_DATA) --proj_output=$(PROJ_PLOTS) \
		--summary_output=$(SUMMARY_PLOTS)

# $(SCORES): scripts/eval.py $(FORECASTS) $(CONFIG)
# 	python $< --pred=$(FORECASTS) --obs=$(RAW_DATA) --config=$(CONFIG) --output=$@


$(FORECASTS): scripts/forecast.py $(RAW_DATA) $(CONFIG)
	python $< --input=$(RAW_DATA) --config=$(CONFIG) --output=$@

$(RAW_DATA): scripts/preprocess.py $(NIS_CACHE) $(CONFIG)
	python $< --cache=$(NIS_CACHE)/clean --config=$(CONFIG) --output=$@

cache: $(NIS_CACHE)/status.txt

$(NIS_CACHE)/status.txt $(TOKEN_PATH):
	python -c "import nisapi; nisapi.cache_all_datasets('$(NIS_CACHE)', '$(TOKEN)')"
	find $(NIS_CACHE)/clean -type f | xargs sha1sum > $@
