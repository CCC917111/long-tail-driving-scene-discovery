PYTHON ?= python
DATA ?= dataset/nuscenes
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
PORT ?= 8765
MANIFEST ?= $(OUTPUT)/protocol/manifest.json
REVIEW_K ?=
NU_VERSION ?= v1.0-mini
PREPARED ?= $(OUTPUT)/prepared
REVIEWS ?=
MODEL ?= models/Cosmos-Reason1-7B
MODEL_REVISION ?=
TOKENS ?= $(PREPARED)/selected_samples.json
MAX_SAMPLES ?= 1
ARCHIVE ?= dataset/downloads/v1.0-mini.tgz
EPOCHS ?= 10
HIDDEN_DIM ?= 256
TOP_K ?= 32
SEED ?= 42
CPU_WEIGHTS ?= models/resnet18/resnet18-f37072fd.pth
CPU_RUN ?= $(OUTPUT)/comparison_runs/sae_abstopk_norm
FRAMES_PER_SCENE ?= 8
PARTS ?=
SAE_DATA = --features $(EXTRACT)/layer$(LAYER)_mlp_output_$(POOLING).npy \
	--meta $(EXTRACT)/meta.json --labels $(LABELS)
EVAL_ARGS = --manifest $(MANIFEST) --review-k $(REVIEW_K)

.PHONY: install label extract manifest train train-no-reward baselines ablations \
	explore screen test lint clean require-budget demo e2e preflight prepare-real import-labels \
	acquire-mini real-validate real-train notebook acquire-cpu-encoder extract-cpu \
	cpu-validate cpu-train cpu-notebook cpu-screen cpu-explore import-mini-parts

import-mini-parts:
	@test -n "$(PARTS)" || (echo 'Set PARTS to all uploaded driving-mini-part ZIP paths'; exit 1)
	$(PYTHON) scripts/import_mini_parts.py --parts $(PARTS) --data-root $(DATA) \
		--report $(OUTPUT)/upload_import.json

acquire-cpu-encoder:
	$(PYTHON) scripts/acquire_cpu_encoder.py --output $(CPU_WEIGHTS)

extract-cpu:
	$(PYTHON) scripts/extract_cpu.py --prepared-dir $(PREPARED) --data-root $(DATA) \
		--weights $(CPU_WEIGHTS) --output-dir $(EXTRACT) --max-samples $(MAX_SAMPLES)

cpu-validate:
	$(PYTHON) scripts/run_real_driving.py validate --feature-protocol cpu-resnet18 \
		--prepared-dir $(PREPARED) --annotations-dir $(OUTPUT)/annotations \
		--extract-dir $(EXTRACT) --layer 1 --pooling mean --output $(OUTPUT)/handoff_validation.json

cpu-train: require-budget
	$(PYTHON) scripts/run_real_driving.py train --feature-protocol cpu-resnet18 \
		--prepared-dir $(PREPARED) --annotations-dir $(OUTPUT)/annotations \
		--extract-dir $(EXTRACT) --layer 1 --pooling mean --output-dir $(OUTPUT)/comparison_runs \
		--review-k $(REVIEW_K) --seed $(SEED) --epochs $(EPOCHS) \
		--hidden-dim $(HIDDEN_DIM) --k $(TOP_K)

cpu-notebook:
	$(PYTHON) scripts/build_driving_notebook.py --feature-protocol cpu-resnet18 \
		--output notebooks/driving_cpu_from_zero.ipynb

cpu-screen:
	$(PYTHON) scripts/screen.py --run-dir $(CPU_RUN) \
		--features $(FEATURES)/layer1_mlp_output_mean.npy --meta $(FEATURES)/meta.json \
		--output-dir $(SCREENING) --device cpu

cpu-explore:
	$(PYTHON) scripts/neuron_explorer.py serve --run $(CPU_RUN) \
		--samples-root $(DATA)/samples --port $(PORT)

acquire-mini:
	$(PYTHON) scripts/acquire_nuscenes_mini.py --archive $(ARCHIVE) --data-root $(DATA) \
		--report $(OUTPUT)/acquisition.json

real-validate:
	$(PYTHON) scripts/run_real_driving.py validate --prepared-dir $(PREPARED) \
		--annotations-dir $(OUTPUT)/annotations --extract-dir $(EXTRACT) \
		--layer $(LAYER) --pooling $(POOLING) --output $(OUTPUT)/handoff_validation.json

real-train: require-budget
	$(PYTHON) scripts/run_real_driving.py train --prepared-dir $(PREPARED) \
		--annotations-dir $(OUTPUT)/annotations --extract-dir $(EXTRACT) \
		--layer $(LAYER) --pooling $(POOLING) --output-dir $(OUTPUT)/comparison_runs \
		--review-k $(REVIEW_K) --seed $(SEED) --epochs $(EPOCHS) \
		--hidden-dim $(HIDDEN_DIM) --k $(TOP_K)

notebook:
	$(PYTHON) scripts/build_driving_notebook.py

preflight:
	$(PYTHON) scripts/driving_preflight.py --data-root $(DATA) --version $(NU_VERSION) \
		--model-dir $(MODEL) --output $(OUTPUT)/from_zero/preflight.json

prepare-real:
	$(PYTHON) scripts/prepare_driving.py prepare --data-root $(DATA) --version $(NU_VERSION) \
		--frames-per-scene $(FRAMES_PER_SCENE) --output-dir $(PREPARED)

import-labels:
	@test -n "$(REVIEWS)" || (echo 'Set REVIEWS to the completed manual_review.csv'; exit 1)
	$(PYTHON) scripts/prepare_driving.py import-labels --prepared-dir $(PREPARED) \
		--reviews $(REVIEWS) --output-dir $(OUTPUT)/annotations

require-budget:
	@test -n "$(REVIEW_K)" || (echo 'Set REVIEW_K before inspecting test results'; exit 1)

manifest: require-budget
	$(PYTHON) scripts/sae_abstopk_tail_reward.py $(SAE_DATA) \
		--review-k $(REVIEW_K) --freeze-manifest-only --output-dir $(dir $(MANIFEST))

install:
	$(PYTHON) -m pip install -r requirements.txt

label:
	$(PYTHON) scripts/annotate_normal_core.py \
		--samples-root $(DATA)/samples \
		--meta-tgz $(DATA)/v1.0-trainval_meta.tgz \
		--output $(LABELS)

extract:
	@test -n "$(MODEL_REVISION)" || (echo 'Set MODEL_REVISION to the pinned model commit'; exit 1)
	$(PYTHON) scripts/extract.py \
		--samples-root $(DATA)/samples \
		--meta-dir $(DATA)/$(NU_VERSION) \
		--model-dir $(MODEL) --model-revision $(MODEL_REVISION) \
		--sample-tokens $(TOKENS) --max-samples $(MAX_SAMPLES) \
		--layers $(LAYERS) \
		--output-dir $(EXTRACT)

train: require-budget
	$(PYTHON) scripts/sae_abstopk_tail_reward.py \
		--features $(EXTRACT)/layer$(LAYER)_mlp_output_$(POOLING).npy \
		--meta $(EXTRACT)/meta.json \
		--labels $(LABELS) \
		$(EVAL_ARGS) --output-dir $(RUN)

train-no-reward: require-budget
	$(PYTHON) scripts/sae_abstopk_tail_reward.py $(SAE_DATA) $(EVAL_ARGS) \
		--reward none --output-dir $(OUTPUT)/sae_no_reward

baselines: require-budget
	$(PYTHON) scripts/supervised_baselines.py --method logistic \
		$(SAE_DATA) $(EVAL_ARGS) --output-dir $(OUTPUT)/logistic
	$(PYTHON) scripts/supervised_baselines.py --method mlp \
		$(SAE_DATA) $(EVAL_ARGS) --output-dir $(OUTPUT)/mlp

ablations: require-budget
	$(PYTHON) scripts/ablation_topk_sae.py \
		--features $(EXTRACT)/layer$(LAYER)_mlp_output_$(POOLING).npy \
		--meta $(EXTRACT)/meta.json --labels $(LABELS) \
		$(EVAL_ARGS) --output-dir $(OUTPUT)/ablation_topk
	$(PYTHON) scripts/ablation_sae_cosmos_baseline.py \
		--features $(EXTRACT)/layer$(LAYER)_mlp_output_$(POOLING).npy \
		--meta $(EXTRACT)/meta.json --labels $(LABELS) \
		$(EVAL_ARGS) --output-dir $(OUTPUT)/ablation_pre_selection

demo:
	$(PYTHON) scripts/synthetic_demo.py --output-dir $(OUTPUT)/synthetic_demo

e2e:
	$(PYTHON) scripts/synthetic_demo.py --output-dir $(OUTPUT)/synthetic_e2e --http

explore:
	$(PYTHON) scripts/neuron_explorer.py serve \
		--run $(RUN) \
		--samples-root $(DATA)/samples \
		--glossary $(GLOSSARY) \
		--port $(PORT)

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
