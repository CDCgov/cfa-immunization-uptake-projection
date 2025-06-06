NICKNAME = test_run
TOKEN_PATH = scripts/socrata_app_token.txt
TOKEN = $(shell cat $(TOKEN_PATH))
CONFIG = scripts/config_template.yaml
RAW_DATA = output/data/$(NICKNAME)/
MODEL_FITS = output/fits/$(NICKNAME)/
DIAGNOSTICS = output/diagnostics/$(NICKNAME)/
PREDICTIONS = output/forecasts/$(NICKNAME)/
SCORES = output/scores/tables/$(NICKNAME)/


.PHONY: clean nis delete_nis viz

all: $(RAW_DATA) $(MODEL_FITS) $(DIAGNOSTICS) $(PREDICTIONS) $(SCORES)

viz:
	streamlit run scripts/viz.py

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

clean:
	rm -r $(RAW_DATA) $(MODEL_FITS) $(DIAGNOSTICS) $(PREDICTIONS) $(SCORES)

nis:
	python -c "import nisapi"
	python -m nisapi cache --app-token=$(TOKEN)

delete_nis:
	python -c "import nisapi"
	python -m nisapi delete
