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

PLOT_DATA = $(OUTPUT_DIR)/plots/data_one_season_by_state.png
PLOT_FORECAST = $(OUTPUT_DIR)/plots/forecast_example.png


.PHONY: clean viz

all: $(CONFIG_COPY) $(DATA) $(FITS) $(DIAGNOSTICS) $(FORECASTS) $(SCORES) $(PLOT_DATA) $(PLOT_FORECAST)

viz:
	streamlit run scripts/viz.py -- \
		--data=$(DATA) --forecasts=$(FORECASTS) --scores=$(SCORES) --config=$(CONFIG)

$(SCORES): scripts/eval.py $(FORECASTS) $(DATA)
	python $< --forecasts=$(FORECASTS) --data=$(DATA) --config=$(CONFIG) --output=$@

$(PLOT_FORECAST): scripts/plot_forecast.py $(CONFIG) $(DATA) $(FORECASTS) $(SCORES)
	python $< --config=$(CONFIG) --data=$(DATA) --forecasts=$(FORECASTS) --scores=$(SCORES) --output=$@

$(FORECASTS): scripts/forecast.py $(DATA) $(FITS) $(CONFIG)
	python $< --data=$(DATA) --fits=$(FITS) --config=$(CONFIG) --output=$@

$(DIAGNOSTICS): scripts/diagnostics.py $(FITS) $(CONFIG)
	python $< --fits=$(FITS) --config=$(CONFIG) --output=$@

$(FITS): scripts/fit.py $(DATA) $(CONFIG)
	python $< --data=$(DATA) --config=$(CONFIG) --output=$@

$(PLOT_DATA): scripts/plot_data.py $(DATA)
	python $< --config=$(CONFIG) --data=$(DATA) --output=$@

$(DATA): scripts/preprocess.py $(RAW_DATA) $(CONFIG)
	python $< --config=$(CONFIG) --input=$(RAW_DATA) --output=$@

$(CONFIG_COPY): $(CONFIG)
	mkdir -p $(OUTPUT_DIR)
	cp $(CONFIG) $@

clean:
	rm -rf $(OUTPUT_DIR)
