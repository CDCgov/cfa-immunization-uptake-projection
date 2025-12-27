RUN_ID = test
TOKEN_PATH = scripts/socrata_app_token.txt
TOKEN = $(shell cat $(TOKEN_PATH))
CONFIG = scripts/config.yaml
SETTINGS = output/settings/$(RUN_ID)/
DATA = output/data/$(RUN_ID)/nis.parquet
FITS = output/fits/$(RUN_ID)/
DIAGNOSTICS = output/diagnostics/$(RUN_ID)/
FORECASTS = output/forecasts/$(RUN_ID)/
SCORES = output/scores/$(RUN_ID)/


.PHONY: clean nis delete_nis viz

all: $(SETTINGS) $(DATA) $(FITS) $(DIAGNOSTICS) $(FORECASTS) $(SCORES)

viz:
	streamlit run scripts/viz.py -- \
	--obs=$(DATA) --pred=$(FORECASTS) --score=$(SCORES) --config=$(CONFIG)

$(SCORES): scripts/eval.py $(FORECASTS) $(DATA)
	python $< \
		--pred=$(FORECASTS) --obs=$(DATA) --config=$(CONFIG) \
		--output=$@

$(FORECASTS): scripts/forecast.py $(DATA) $(FITS) $(CONFIG)
	python $< --data=$(DATA) --models=$(FITS) --config=$(CONFIG) \
	--output=$@

$(DIAGNOSTICS): scripts/diagnostics.py $(FITS) $(CONFIG)
	python $< --input=$(FITS) --config=$(CONFIG) --output=$@

$(FITS): scripts/fit.py $(DATA) $(CONFIG)
	python $< --data=$(DATA) --config=$(CONFIG) --output=$@

$(DATA): scripts/preprocess.py $(CONFIG)
	python $< --config=$(CONFIG) --output=$@

$(SETTINGS): $(CONFIG)
	mkdir -p $(SETTINGS)
	cp $(CONFIG) $(SETTINGS)

clean:
	rm -r $(SETTINGS) $(DATA) $(FITS) $(DIAGNOSTICS) $(FORECASTS) $(SCORES)

nis:
	python -c "import nisapi"
	python -m nisapi cache --app-token=$(TOKEN)

delete_nis:
	python -c "import nisapi"
	python -m nisapi delete
