RUN_ID = test

RAW_DATA = data/raw.parquet
CONFIG = scripts/config.yaml

OUTPUT_DIR = output/$(RUN_ID)
CONFIG_COPY = $(OUTPUT_DIR)/config.yaml
DATA = $(OUTPUT_DIR)/nis.parquet
FITS = $(OUTPUT_DIR)/fits.pkl
PREDS = $(OUTPUT_DIR)/predictions.parquet
DIAGNOSTICS = $(OUTPUT_DIR)/diagnostics/status.txt
SCORES = $(OUTPUT_DIR)/scores.parquet

PLOT_DATA = $(OUTPUT_DIR)/plots/data_one_season_by_state.png
PLOT_PREDS = $(OUTPUT_DIR)/plots/forecast_example.png


.PHONY: clean viz

all: $(CONFIG_COPY) $(DATA) $(FITS) $(DIAGNOSTICS) $(PREDS) $(SCORES) $(PLOT_DATA) $(PLOT_FORECAST)

viz:
	streamlit run scripts/viz.py -- \
		--data=$(DATA) --preds=$(PREDS) --scores=$(SCORES) --config=$(CONFIG)

$(SCORES): scripts/eval.py $(PREDS) $(DATA)
	python $< --preds=$(PREDS) --data=$(DATA) --config=$(CONFIG) --output=$@

$(PLOT_PREDS): scripts/plot_pred.py $(CONFIG) $(DATA) $(PREDS) $(SCORES)
	python $< --config=$(CONFIG) --data=$(DATA) --preds=$(PREDS) --scores=$(SCORES) --output=$@

$(PREDS): scripts/predict.py $(DATA) $(FITS) $(CONFIG)
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
