SHELL := /bin/bash
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PY ?= $(if $(wildcard $(ROOT)/.venv/bin/python3),$(ROOT)/.venv/bin/python3,python3.11)
GROUPID ?=
REQUIRE_GRAD ?= 0
THRESHOLD ?= 0.1

export SAFE_ACTUATION_ROOT := $(ROOT)
export PYTHONDONTWRITEBYTECODE := 1

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show every command and what it requires
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

.PHONY: submodules
submodules: ## Setup only: initialize exact OpenPI and SST pins, no model weights
	@git -C $(ROOT) submodule update --init --recursive

.PHONY: setup
setup: ## Setup only: create the Python environment on a CPU compute node, no model weights
	@$(ROOT)/scripts/setup.sh

.PHONY: doctor
doctor: ## Read-only: check host, Python version, submodule pins, and required tool availability
	@$(PY) $(ROOT)/scripts/doctor.py

.PHONY: build
build: ## CPU compute: build the reproducible Apptainer image, no model weights
	@$(ROOT)/scripts/build_container.sh

.PHONY: test-offline
test-offline: ## Fixture only: run the public model-free tests, no SST or GPU
	@PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q \
		-m "not live_openpi and not live_sst and not sol_validation"

.PHONY: test-baseline
test-baseline: ## Starter-safe: test supplied infrastructure and Part 1 without graded TODOs
	@PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q \
		tests/test_captured.py \
		tests/test_context.py \
		tests/test_fixtures.py \
		tests/test_openpi_adapter.py \
		tests/test_protection.py \
		tests/test_records.py \
		tests/test_setup.py \
		tests/test_simulator.py \
		tests/test_sst_adapter.py

.PHONY: test-acceptance
test-acceptance: ## Exercises: test all five graded TODO regions; starter failures are intentional
	@PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q \
		tests/test_motion.py \
		tests/test_actuator.py \
		tests/test_enforcement.py

.PHONY: test-sst
test-sst: ## SST: test the pinned Python API and course live integration, Auth state required for marked tests
	@$(ROOT)/scripts/test_sst.sh

.PHONY: inspect-actions
inspect-actions: ## Part 1: report the available pi0-FAST response, its tokens, decode status, and shapes
	@$(PY) -m safe_agent_actuation.cli inspect-actions

.PHONY: low-risk
low-risk: ## Fixture: run the trusted low-risk path in the recording actuator harness, no SST
	@$(PY) -m safe_agent_actuation.cli scenario low-risk --threshold $(THRESHOLD)

.PHONY: high-risk-allow
high-risk-allow: ## Fixture plus real SST: authorized high-risk path with allowed context, Auth required
	@$(PY) -m safe_agent_actuation.cli scenario high-risk-allow --threshold $(THRESHOLD)

.PHONY: high-risk-deny
high-risk-deny: ## Fixture plus real SST: deny high-risk action and hold before actuation, Auth required
	@$(PY) -m safe_agent_actuation.cli scenario high-risk-deny --threshold $(THRESHOLD)

.PHONY: bypass
bypass: ## Fixture plus real SST: direct unauthenticated actuator attempt must be rejected, Auth required
	@$(PY) -m safe_agent_actuation.cli scenario bypass --threshold $(THRESHOLD)

.PHONY: high-risk-allow-offline
high-risk-allow-offline: ## Offline diagnostic: authorized high-risk path without SST, labeled offline
	@$(PY) -m safe_agent_actuation.cli scenario high-risk-allow --offline-diagnostic --threshold $(THRESHOLD)

.PHONY: high-risk-deny-offline
high-risk-deny-offline: ## Offline diagnostic: denied high-risk path without SST, labeled offline
	@$(PY) -m safe_agent_actuation.cli scenario high-risk-deny --offline-diagnostic --threshold $(THRESHOLD)

.PHONY: bypass-offline
bypass-offline: ## Offline diagnostic: in-process bypass rejection without SST, labeled offline
	@$(PY) -m safe_agent_actuation.cli scenario bypass --offline-diagnostic --threshold $(THRESHOLD)

.PHONY: threshold-sweep
threshold-sweep: ## Part 4 real SST: fixed-pair multi-threshold evaluation and measured latency, Auth required
	@$(PY) -m safe_agent_actuation.cli threshold-sweep

.PHONY: threshold-sweep-offline
threshold-sweep-offline: ## Offline diagnostic: fixed-pair policy model, explicitly not live SST
	@$(PY) -m safe_agent_actuation.cli threshold-sweep --offline-diagnostic

.PHONY: grad-key-validity
grad-key-validity: ## CSE 598 real SST: exact 5/20/40-second validity periods and context-change windows, Auth required
	@$(ROOT)/sst/scripts/key_validity.sh

.PHONY: grad-key-validity-offline
grad-key-validity-offline: ## CSE 598 offline diagnostic: session-key reuse and expiration on a manual clock, no SST
	@$(PY) -m safe_agent_actuation.cli key-validity

.PHONY: build-auth
build-auth: ## SST setup: build Auth from the clean pinned submodule on a CPU compute node
	@$(ROOT)/sst/scripts/build_auth.sh

.PHONY: generate-sst
generate-sst: ## SST setup: generate credentials and Auth state only under runtime/sst
	@$(ROOT)/sst/scripts/generate_runtime.sh

.PHONY: auth-start
auth-start: ## SST: start the generated Auth process on a compute node
	@$(ROOT)/sst/scripts/auth.sh start

.PHONY: auth-stop
auth-stop: ## SST: stop only the Auth PID recorded under runtime/sst
	@$(ROOT)/sst/scripts/auth.sh stop

.PHONY: submission
submission: ## Packaging: build one allowlisted ZIP; CSE 598 also sets REQUIRE_GRAD=1
	@test -n "$(GROUPID)" || { echo "usage: make submission GROUPID=<groupid> [REQUIRE_GRAD=1]" >&2; exit 2; }
	@test "$(REQUIRE_GRAD)" = "0" -o "$(REQUIRE_GRAD)" = "1" || { echo "REQUIRE_GRAD must be 0 or 1" >&2; exit 2; }
	@$(PY) $(ROOT)/scripts/validate_results.py $(if $(filter 1,$(REQUIRE_GRAD)),--require-grad,) $(ROOT)/results
	@$(PY) $(ROOT)/scripts/make_submission.py --groupid "$(GROUPID)" $(if $(filter 1,$(REQUIRE_GRAD)),--require-grad,)

.PHONY: lint
lint: ## Offline: run Ruff on the course-maintained Python code
	@$(PY) -m ruff check .

.PHONY: clean
clean: ## DELETES generated results, runtime SST secrets/state, caches, and local environments
	@$(ROOT)/scripts/clean.sh
