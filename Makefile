RUN_ID = test
CONFIG = scripts/config.yaml
SETTINGS = output/settings/$(RUN_ID)/
RAW_DATA = data/raw.parquet
DATA = output/data/$(RUN_ID)/nis.parquet
FITS = output/fits/$(RUN_ID)/
DIAGNOSTICS = output/diagnostics/$(RUN_ID)/
FORECASTS = output/forecasts/$(RUN_ID)/
SCORES = output/scores/$(RUN_ID)/
DATA_PLOT = output/diagnostics/$(RUN_ID)/data_national.png


.PHONY: clean viz

all: $(SETTINGS) $(DATA) $(FITS) $(DIAGNOSTICS) $(FORECASTS) $(SCORES) $(DATA_PLOT)

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

$(DATA_PLOT): scripts/describe_data.py $(DATA)
	python $< --input=$(DATA) --output_dir=output/diagnostics/$(RUN_ID)/

$(DATA): scripts/preprocess.py $(RAW_DATA) $(CONFIG)
	python $< --config=$(CONFIG) --input=$(RAW_DATA) --output=$@

$(SETTINGS): $(CONFIG)
	mkdir -p $(SETTINGS)
	cp $(CONFIG) $(SETTINGS)

clean:
	rm -r $(SETTINGS) $(DATA) $(FITS) $(DIAGNOSTICS) $(FORECASTS) $(SCORES)
