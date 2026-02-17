CONFIG = scripts/config_vignette.yaml
RAW_DATA = data/raw.parquet

RUN_ID = $(shell python scripts/get_run_id.py --config=$(CONFIG))

OUTPUT_DIR = output/$(RUN_ID)
CONFIG_COPY = $(OUTPUT_DIR)/config.yaml
DATA = $(OUTPUT_DIR)/data.parquet
PRED_DIR = $(OUTPUT_DIR)/pred
SCORES = $(OUTPUT_DIR)/scores.parquet

# each plotting script outputs multiple files; use a single output as a flag
PLOT_DATA = $(OUTPUT_DIR)/plots/.data.checkpoint
PLOT_PREDS = $(OUTPUT_DIR)/plots/.preds.checkpoint
PLOT_SCORES = $(OUTPUT_DIR)/plots/.scores.checkpoint

# dynamically find forecast dates to use
FORECAST_DATES = $(shell python scripts/get_forecast_dates.py --config=$(CONFIG))
PREDS = $(foreach date,$(FORECAST_DATES),$(PRED_DIR)/forecast_date=$(date)/part-0.parquet)
FITS = $(foreach date,$(FORECAST_DATES),$(OUTPUT_DIR)/fits/fit_$(date).pkl)

# This variable because the pattern `forecast_date=2020-01-01` confuses make.
# It thinks `=%` is variable assignment, not pattern matching.
# So we need `forecast_date$(EQ)%`.
EQ = =

.PHONY: clean viz dx

all: $(CONFIG_COPY) $(PLOT_DATA) $(PLOT_PREDS) $(PLOT_SCORES)

viz:
	streamlit run scripts/viz.py -- \
		--data=$(DATA) --preds=$(PRED_DIR) --scores=$(SCORES) --config=$(CONFIG)

dx: scripts.diagnostics $(FITS) $(CONFIG)
	python $< --fit_dir=$(OUTPUT_DIR)/fits --output_dir=$(OUTPUT_DIR)/diagnostics --config=$(CONFIG)

$(PLOT_SCORES): scripts/plot_scores.py $(SCORES) $(CONFIG)
	python $< --scores=$(SCORES) --config=$(CONFIG) --output=$@

$(PLOT_PREDS): scripts/plot_preds.py $(CONFIG) $(DATA) $(PREDS)
	python $< --config=$(CONFIG) --data=$(DATA) --preds=$(PRED_DIR) --output=$@

$(PLOT_DATA): scripts/plot_data.py $(DATA) $(CONFIG)
	python $< --config=$(CONFIG) --data=$(DATA) --output=$@

$(SCORES): scripts/eval.py $(PREDS) $(DATA) $(CONFIG)
	python $< --preds=$(PRED_DIR) --data=$(DATA) --config=$(CONFIG) --output=$@

# output/run_id/pred/forecast_date=2021-01-01/part-0.parquet <== output/fits/fit_2021-01-01.pkl
$(PRED_DIR)/forecast_date$(EQ)%/part-0.parquet: scripts/predict.py $(OUTPUT_DIR)/fits/fit_%.pkl
	python $< --fits=$(OUTPUT_DIR)/fits/fit_$*.pkl --output=$@

$(OUTPUT_DIR)/fits/fit_%.pkl: scripts/fit.py $(DATA) $(CONFIG)
	python $< --data=$(DATA) --forecast_date=$* --config=$(CONFIG) --output=$@

$(DATA): scripts/preprocess.py $(RAW_DATA) $(CONFIG)
	python $< --config=$(CONFIG) --input=$(RAW_DATA) --output=$@

$(CONFIG_COPY): $(CONFIG)
	mkdir -p $(OUTPUT_DIR)
	cp $(CONFIG) $@

clean:
	rm -rf $(OUTPUT_DIR)
