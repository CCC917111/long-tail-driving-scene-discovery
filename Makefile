PYTHON ?= python
DATA ?= data/nuscenes
OUTPUT ?= output
LAYERS ?= 28
LAYER ?= 28
POOLING ?= mean
LABELS ?= $(OUTPUT)/annotations/labels.json
EXTRACT ?= $(OUTPUT)/extract
FEATURES ?= $(EXTRACT)
RUN ?= $(OUTPUT)/sae_abstopk_tail_reward
GLOSSARY ?= results/neuron_glossary.csv
SCREENING ?= $(OUTPUT)/screening

.PHONY: install label extract train ablations screen test lint clean

install:
	$(PYTHON) -m pip install -r requirements.txt

label:
	$(PYTHON) scripts/annotate_normal_core.py \
		--samples-root $(DATA)/samples \
		--meta-tgz $(DATA)/v1.0-trainval_meta.tgz \
		--output $(LABELS)

extract:
	$(PYTHON) scripts/extract.py \
		--samples-root $(DATA)/samples \
		--meta-tgz $(DATA)/v1.0-trainval_meta.tgz \
		--sample-tokens $(LABELS) \
		--layers $(LAYERS) \
		--output-dir $(EXTRACT)

train:
	$(PYTHON) scripts/sae_abstopk_tail_reward.py \
		--features $(EXTRACT)/layer$(LAYER)_mlp_output_$(POOLING).npy \
		--meta $(EXTRACT)/meta.json \
		--labels $(LABELS) \
		--output-dir $(RUN)

ablations:
	$(PYTHON) scripts/ablation_topk_sae.py \
		--features $(EXTRACT)/layer$(LAYER)_mlp_output_$(POOLING).npy \
		--meta $(EXTRACT)/meta.json --labels $(LABELS) \
		--output-dir $(OUTPUT)/ablation_topk
	$(PYTHON) scripts/ablation_sae_cosmos_baseline.py \
		--features $(EXTRACT)/layer$(LAYER)_mlp_output_$(POOLING).npy \
		--meta $(EXTRACT)/meta.json --labels $(LABELS) \
		--output-dir $(OUTPUT)/ablation_pre_selection

screen:
	$(PYTHON) scripts/screen.py \
		--run-dir $(RUN) \
		--features $(FEATURES)/layer$(LAYER)_mlp_output_$(POOLING).npy \
		--meta $(FEATURES)/meta.json \
		--glossary $(GLOSSARY) \
		--output-dir $(SCREENING)

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m flake8 scripts tests --max-line-length 100

clean:
	rm -rf .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
