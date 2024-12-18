NIS_CACHE = .cache/nisapi
TOKEN_PATH = scripts/socrata_app_token.txt
TOKEN = $(shell cat $(TOKEN_PATH))
CONFIG = scripts/config.yaml

.PHONY: cache

run: $(CONFIG) cache
	python scripts/main.py --config=$(CONFIG) --cache=$(NIS_CACHE)/clean

data/scores.parquet: scripts/eval.py
	echo MISSING STUFF

data/forecasts.parquet: scripts/forecast.py
	echo MISSING STUFF

data/nis_raw.parquet: scripts/preprocess.py cache
	python $< --cache=$(NIS_CACHE)/clean --output=$@

cache: $(NIS_CACHE)/status.txt

$(NIS_CACHE)/status.txt $(TOKEN_PATH):
	python -c "import nisapi; nisapi.cache_all_datasets('$(NIS_CACHE)', '$(TOKEN)')"
	find $(NIS_CACHE)/clean -type f | xargs sha1sum > $@
