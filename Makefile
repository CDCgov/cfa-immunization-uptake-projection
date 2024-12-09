NIS_CACHE = .cache/nisapi
TOKEN_PATH = scripts/socrata_app_token.txt
TOKEN = $(shell cat $(TOKEN_PATH))
CONFIG = scripts/config.yaml

run: $(CONFIG)
	python scripts/main.py --config=$(CONFIG)

$(NIS_CACHE)/status.txt $(TOKEN_PATH):
	python -c "import nisapi; nisapi.cache_all_datasets('$(NIS_CACHE)', '$(TOKEN)')"
	find $(NIS_CACHE)/clean -type f | xargs sha1sum > $@
