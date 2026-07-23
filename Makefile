.PHONY: help sync lint format test run ask show-schema \
        memgraph-up memgraph-down memgraph-logs memgraph-cli clean \
        cypherbench-download cypherbench-eval cypherbench-eval-single cypherbench-eval-agentic \
        cypherbench-download-domain run-single run-agentic recap site validate-results

UV := env -u VIRTUAL_ENV uv

MEMGRAPH_CONTAINER := multigraphrag-memgraph
MEMGRAPH_IMAGE := memgraph/memgraph-mage:latest
MEMGRAPH_BOLT_PORT := 7687
MEMGRAPH_HTTP_PORT := 7444
MEMGRAPH_LAB_PORT := 3000

# CypherBench evaluation defaults, override on the command line, e.g.:
#   make cypherbench-eval CYPHERBENCH_SPLIT=test CYPHERBENCH_DOMAINS=geography CYPHERBENCH_LIMIT=50
CYPHERBENCH_DIR := data/cypherbench
CYPHERBENCH_SPLIT := test
CYPHERBENCH_DOMAINS :=
CYPHERBENCH_LIMIT :=
CYPHERBENCH_MODE := both
CYPHERBENCH_GRAPH_VARIANT := simplekg_sampled
CYPHERBENCH_CONCURRENCY :=
CYPHERBENCH_TRACE := $(CYPHERBENCH_DIR)/trace_$(CYPHERBENCH_SPLIT)_$(CYPHERBENCH_MODE).jsonl

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sync: ## Install/sync project dependencies with uv
	$(UV) sync

lint: ## Run ruff checks
	$(UV) run ruff check src tests

format: ## Auto-format and fix lint issues with ruff
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

test: ## Run the test suite
	$(UV) run pytest

ask: ## Run `multigraphrag ask "<question>"`, e.g. make ask Q="how many doors are there?"
	$(UV) run multigraphrag ask "$(Q)"

show-schema: ## Print the graph schema as injected into the Query Generator prompt
	$(UV) run multigraphrag show-schema

memgraph-up: ## Start a local Memgraph instance via Docker (Bolt on :7687)
	docker run -d --rm \
		--name $(MEMGRAPH_CONTAINER) \
		-p $(MEMGRAPH_BOLT_PORT):7687 \
		-p $(MEMGRAPH_HTTP_PORT):7444 \
		$(MEMGRAPH_IMAGE)
	@echo "Memgraph is starting. Bolt endpoint: bolt://localhost:$(MEMGRAPH_BOLT_PORT)"

memgraph-down: ## Stop and remove the local Memgraph container
	docker stop $(MEMGRAPH_CONTAINER)

memgraph-logs: ## Tail logs from the local Memgraph container
	docker logs -f $(MEMGRAPH_CONTAINER)

memgraph-cli: ## Open an interactive Cypher shell against the local Memgraph instance
	docker exec -it $(MEMGRAPH_CONTAINER) mgconsole

cypherbench-download: ## Download CypherBench (train+test tasks + sampled graphs) from HuggingFace, e.g. add CYPHERBENCH_DOMAINS=geography,art
	$(UV) run multigraphrag cypherbench download --dest $(CYPHERBENCH_DIR) \
		--graph-variant $(CYPHERBENCH_GRAPH_VARIANT) \
		$(if $(CYPHERBENCH_DOMAINS),--domains $(CYPHERBENCH_DOMAINS),)

cypherbench-eval: ## Run Single-vs-Agentic comparison, e.g. make cypherbench-eval CYPHERBENCH_SPLIT=test CYPHERBENCH_DOMAINS=geography CYPHERBENCH_LIMIT=50 CYPHERBENCH_MODE=agentic CYPHERBENCH_CONCURRENCY=5
	$(UV) run multigraphrag cypherbench evaluate --dest $(CYPHERBENCH_DIR) \
		--split $(CYPHERBENCH_SPLIT) --mode $(CYPHERBENCH_MODE) \
		--graph-variant $(CYPHERBENCH_GRAPH_VARIANT) \
		--trace $(CYPHERBENCH_TRACE) \
		$(if $(CYPHERBENCH_DOMAINS),--domains $(CYPHERBENCH_DOMAINS),) \
		$(if $(CYPHERBENCH_LIMIT),--limit $(CYPHERBENCH_LIMIT),) \
		$(if $(CYPHERBENCH_CONCURRENCY),--concurrency $(CYPHERBENCH_CONCURRENCY),)

cypherbench-eval-single: ## Shortcut: only the linear-pass baseline (no self-correction loop)
	$(MAKE) cypherbench-eval CYPHERBENCH_MODE=single

cypherbench-eval-agentic: ## Shortcut: only the full Multi-Agent GraphRAG pipeline
	$(MAKE) cypherbench-eval CYPHERBENCH_MODE=agentic

# --- Reproducible per-model/per-domain runs -> results/ --------------------
# One run-single/run-agentic invocation always covers exactly one domain and
# one mode, writing a self-contained leaf directory under results/:
#   trace.jsonl   -- one row per question (gold/generated Cypher, answer, correct)
#   calls.jsonl   -- every LLM call made, tagged with the question's qid
#   run.json      -- the settings used (model/temperature/reasoning/...) plus
#                    the resulting accuracy, so it never needs to be
#                    reconstructed by parsing file/directory names
# This is the "official", publishable counterpart to the ad-hoc
# cypherbench-eval target above, meant to be committed to results/ via a PR.
#
# Examples:
#   make cypherbench-download-domain DOMAIN=geography
#   make run-single  MODEL=Qwen/Qwen3.5-27B DOMAIN=geography
#   make run-agentic MODEL=Qwen/Qwen3.5-27B DOMAIN=geography TEMP=1.0 REASONING=off
CYPHERBENCH_TRAIN_ONLY_DOMAINS := art biology soccer terrorist_attack

RESULTS_DIR := results
MODEL :=
DOMAIN :=
TEMP := 0.0
REASONING := medium
LIMIT := 40
RUN_CONCURRENCY := 2
RUN_GRAPH_VARIANT := simplekg

MODEL_SLUG := $(subst /,--,$(MODEL))
RUN_SPLIT := $(if $(filter $(DOMAIN),$(CYPHERBENCH_TRAIN_ONLY_DOMAINS)),train,test)
RUN_CONFIG_DIR := $(RESULTS_DIR)/$(MODEL_SLUG)/temp$(TEMP)-reasoning-$(REASONING)/$(DOMAIN)
RUN_ENV = MULTIGRAPHRAG_LLM__MODEL=$(MODEL) MULTIGRAPHRAG_LLM__TEMPERATURE=$(TEMP) $(if $(filter off,$(REASONING)),MULTIGRAPHRAG_LLM__REASONING_ENABLED=false,MULTIGRAPHRAG_LLM__REASONING_ENABLED=true MULTIGRAPHRAG_LLM__REASONING_EFFORT=$(REASONING))

cypherbench-download-domain: ## Download one full-scale CypherBench domain, e.g. make cypherbench-download-domain DOMAIN=geography
	@test -n "$(DOMAIN)" || (echo "DOMAIN is required, e.g. DOMAIN=geography" && exit 1)
	$(UV) run multigraphrag cypherbench download --dest $(CYPHERBENCH_DIR) \
		--domains $(DOMAIN) --graph-variant $(RUN_GRAPH_VARIANT)

run-single: ## Run the single-pass baseline for one model/domain into results/, e.g. make run-single MODEL=Qwen/Qwen3.5-27B DOMAIN=geography
	@test -n "$(MODEL)" || (echo "MODEL is required, e.g. MODEL=Qwen/Qwen3.5-27B" && exit 1)
	@test -n "$(DOMAIN)" || (echo "DOMAIN is required, e.g. DOMAIN=geography" && exit 1)
	$(RUN_ENV) $(UV) run multigraphrag cypherbench evaluate --dest $(CYPHERBENCH_DIR) \
		--split $(RUN_SPLIT) --domains $(DOMAIN) --mode single \
		--graph-variant $(RUN_GRAPH_VARIANT) --concurrency $(RUN_CONCURRENCY) \
		$(if $(LIMIT),--limit $(LIMIT),) \
		--trace $(RUN_CONFIG_DIR)/single/trace.jsonl \
		--call-log $(RUN_CONFIG_DIR)/single/calls.jsonl \
		--run-manifest $(RUN_CONFIG_DIR)/single/run.json

run-agentic: ## Run the full Multi-Agent GraphRAG pipeline for one model/domain into results/, e.g. make run-agentic MODEL=Qwen/Qwen3.5-27B DOMAIN=geography
	@test -n "$(MODEL)" || (echo "MODEL is required, e.g. MODEL=Qwen/Qwen3.5-27B" && exit 1)
	@test -n "$(DOMAIN)" || (echo "DOMAIN is required, e.g. DOMAIN=geography" && exit 1)
	$(RUN_ENV) $(UV) run multigraphrag cypherbench evaluate --dest $(CYPHERBENCH_DIR) \
		--split $(RUN_SPLIT) --domains $(DOMAIN) --mode agentic \
		--graph-variant $(RUN_GRAPH_VARIANT) --concurrency $(RUN_CONCURRENCY) \
		$(if $(LIMIT),--limit $(LIMIT),) \
		--trace $(RUN_CONFIG_DIR)/agentic/trace.jsonl \
		--call-log $(RUN_CONFIG_DIR)/agentic/calls.jsonl \
		--run-manifest $(RUN_CONFIG_DIR)/agentic/run.json

recap: ## Regenerate results/RECAP.md + results/recap.json from all results/**/run.json
	$(UV) run python scripts/build_recap.py

site: ## Regenerate docs/ (static site) from results/, ready for GitHub Pages
	$(UV) run python scripts/build_site.py

validate-results: ## Check results/**/{trace,calls,run}.jsonl|json are well-formed
	$(UV) run python scripts/validate_results.py

clean: ## Remove caches and build artifacts
	rm -rf .venv .ruff_cache .pytest_cache dist build
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
