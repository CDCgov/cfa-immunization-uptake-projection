RUN_ID = test

RAW_DATA = data/raw.parquet
CONFIG = scripts/config.yaml

OUTPUT_DIR = output/$(RUN_ID)
CONFIG_COPY = $(OUTPUT_DIR)/config.yaml
DATA = $(OUTPUT_DIR)/nis.parquet
FITS = $(OUTPUT_DIR)/fits.pkl
FORECASTS = $(OUTPUT_DIR)/forecasts.parquet
DIAGNOSTICS = $(OUTPUT_DIR)/diagnostics/status.txt
SCORES = $(OUTPUT_DIR)/scores.parquet

DATA_PLOT = $(OUTPUT_DIR)/plots/data_national.png


.PHONY: clean viz

all: $(SETTINGS) $(DATA) $(FITS) $(DIAGNOSTICS) $(FORECASTS) $(SCORES) $(DATA_PLOT)

viz:
	streamlit run scripts/viz.py -- \
		--data=$(DATA) --forecasts=$(FORECASTS) --scores=$(SCORES) --config=$(CONFIG)

$(SCORES): scripts/eval.py $(FORECASTS) $(DATA)
	python $< --forecasts=$(FORECASTS) --data=$(DATA) --config=$(CONFIG) --output=$@

$(FORECASTS): scripts/forecast.py $(DATA) $(FITS) $(CONFIG)
	python $< --data=$(DATA) --fits=$(FITS) --config=$(CONFIG) --output=$@

$(DIAGNOSTICS): scripts/diagnostics.py $(FITS) $(CONFIG)
	python $< --fits=$(FITS) --config=$(CONFIG) --output=$@

$(FITS): scripts/fit.py $(DATA) $(CONFIG)
	python $< --data=$(DATA) --config=$(CONFIG) --output=$@

$(DATA_PLOT): scripts/describe_data.py $(DATA)
	python $< --input=$(DATA) --output_dir=$(OUTPUT_DIR)/plots

$(DATA): scripts/preprocess.py $(RAW_DATA) $(CONFIG)
	python $< --config=$(CONFIG) --input=$(RAW_DATA) --output=$@

$(CONFIG_COPY): $(CONFIG)
	mkdir -p $(OUTPUT_DIR)
	cp $(CONFIG) $(CONFIG_COPY)

clean:
	rm -rf $(OUTPUT_DIR)
