RUN_ID = psr
TOKEN_PATH = scripts/socrata_app_token.txt
TOKEN = $(shell cat $(TOKEN_PATH))
CONFIG = scripts/config_psr.yaml
SETTINGS = output/settings/$(RUN_ID)/
RAW_DATA = output/data/$(RUN_ID)/
MODEL_FITS = output/fits/$(RUN_ID)/
DIAGNOSTICS = output/diagnostics/$(RUN_ID)/
PREDICTIONS = output/forecasts/$(RUN_ID)/
SCORES = output/scores/$(RUN_ID)/


.PHONY: clean nis delete_nis viz

all: $(SETTINGS) $(RAW_DATA) $(MODEL_FITS) $(DIAGNOSTICS) $(PREDICTIONS) $(SCORES)

viz:
	streamlit run scripts/viz.py -- \
	--obs=$(RAW_DATA) --pred=$(PREDICTIONS) --score=$(SCORES) --config=$(CONFIG)

$(SCORES): scripts/eval.py $(PREDICTIONS) $(RAW_DATA)
	python $< \
		--pred=$(PREDICTIONS) --obs=$(RAW_DATA) --config=$(CONFIG) \
		--output=$@

$(PREDICTIONS): scripts/forecast.py $(RAW_DATA) $(MODEL_FITS) $(CONFIG)
	python $< --input=$(RAW_DATA) --models=$(MODEL_FITS) --config=$(CONFIG) \
	--output=$@

$(DIAGNOSTICS): scripts/diagnostics.py $(MODEL_FITS) $(CONFIG)
	python $< --input=$(MODEL_FITS) --config=$(CONFIG) --output=$@

$(MODEL_FITS): scripts/fit.py $(RAW_DATA) $(CONFIG)
	python $< --input=$(RAW_DATA) --config=$(CONFIG) --output=$@

$(RAW_DATA): scripts/preprocess.py $(CONFIG)
	python $< --config=$(CONFIG) --output=$@

$(SETTINGS): $(CONFIG)
	mkdir -p $(SETTINGS)
	cp $(CONFIG) $(SETTINGS)

clean:
	rm -r $(SETTINGS) $(RAW_DATA) $(MODEL_FITS) $(DIAGNOSTICS) $(PREDICTIONS) $(SCORES)

nis:
	python -c "import nisapi"
	python -m nisapi cache --app-token=$(TOKEN)

delete_nis:
	python -c "import nisapi"
	python -m nisapi delete
