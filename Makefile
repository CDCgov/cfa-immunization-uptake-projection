RUN_ID = full3

RAW_DATA = data/raw.parquet
CONFIG = scripts/config.yaml

OUTPUT_DIR = output/$(RUN_ID)
CONFIG_COPY = $(OUTPUT_DIR)/config.yaml
DATA = $(OUTPUT_DIR)/data.parquet
PRED_DIR = $(OUTPUT_DIR)/pred
PREDS_FLAG = $(PRED_DIR)/.checkpoint
SCORES = $(OUTPUT_DIR)/scores.parquet

PLOT_DATA = $(OUTPUT_DIR)/plots/data_one_season_by_state.png
PLOT_PREDS = $(OUTPUT_DIR)/plots/forecast_example.png

FORECAST_STARTS = $(shell python scripts/get_forecast_starts.py --config=$(CONFIG))
PREDS = $(foreach date,$(FORECAST_STARTS),$(PRED_DIR)/forecast_start=$(date)/part-0.parquet)
FITS = $(foreach date,$(FORECAST_STARTS),$(OUTPUT_DIR)/fits/fit_$(date).pkl)

# This variable because the pattern `forecast_date=2020-01-01` confuses make.
# It thinks `=%` is variable assignment, not pattern matching.
# So we need `forecast_date$(EQ)%`.
EQ = =

.PHONY: clean viz dx

all: $(CONFIG_COPY) $(DATA) $(FITS) $(PREDS) $(SCORES) $(PLOT_DATA) $(PLOT_PREDS)

viz:
	streamlit run scripts/viz.py -- \
		--data=$(DATA) --preds=$(PRED_DIR) --scores=$(SCORES) --config=$(CONFIG)

dx: scripts.diagnostics $(FITS) $(CONFIG)
	python $< --fit_dir=$(OUTPUT_DIR)/fits --output_dir=$(OUTPUT_DIR)/diagnostics --config=$(CONFIG)

$(SCORES): scripts/eval.py $(PREDS_FLAG) $(DATA) $(CONFIG)
	python $< --preds=$(PRED_DIR) --data=$(DATA) --config=$(CONFIG) --output=$@

$(PLOT_PREDS): scripts/plot_preds.py $(CONFIG) $(DATA) $(PREDS_FLAG) $(SCORES)
	python $< --config=$(CONFIG) --data=$(DATA) --preds=$(PRED_DIR) --scores=$(SCORES) --output=$@

$(PLOT_DATA): scripts/plot_data.py $(DATA)
	python $< --config=$(CONFIG) --data=$(DATA) --output=$@

$(PLOT_DATA): scripts/plot_data.py $(DATA) $(CONFIG)
	python $< --config=$(CONFIG) --data=$(DATA) --output=$@

# output/run_id/pred/forecast_start=2021-01-01/part-0.parquet <== output/fits/fit_2021-01-01.pkl
$(PRED_DIR)/forecast_start$(EQ)%/part-0.parquet: scripts/predict.py $(OUTPUT_DIR)/fits/fit_%.pkl $(DATA) $(CONFIG)
	python $< --data=$(DATA) --fits=$(OUTPUT_DIR)/fits/fit_$*.pkl --config=$(CONFIG) --output=$@

$(OUTPUT_DIR)/fits/fit_%.pkl: scripts/fit.py $(DATA) $(CONFIG)
	python $< --data=$(DATA) --forecast_start=$* --config=$(CONFIG) --output=$@

$(DATA): scripts/preprocess.py $(RAW_DATA) $(CONFIG)
	python $< --config=$(CONFIG) --input=$(RAW_DATA) --output=$@

$(CONFIG_COPY): $(CONFIG)
	mkdir -p $(OUTPUT_DIR)
	cp $(CONFIG) $@

clean:
	rm -rf $(OUTPUT_DIR)
