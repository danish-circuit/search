# Convenience targets for the search-agent demo + its evaluation harness.

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: up
up: ## Build images and start the full stack (db, api, frontend, evaluator)
	docker compose up --build

.PHONY: down
down: ## Stop the stack
	docker compose down

.PHONY: opik
opik: ## Start a local Opik instance (use 'make opik stop' to stop)
	@if echo "$(MAKECMDGOALS)" | grep -q "stop"; then \
		if [ ! -d "opik" ]; then \
			echo "Opik repository not found. Nothing to stop."; \
			exit 0; \
		fi; \
		echo "Stopping Opik..."; \
		cd opik && ./opik.sh --stop; \
	else \
		if [ ! -d "opik" ]; then \
			echo "Cloning Opik repository..."; \
			git clone https://github.com/comet-ml/opik.git; \
		else \
			echo "Opik repository exists, pulling latest changes..."; \
			(cd opik && git pull); \
		fi; \
		echo "Starting Opik (UI at http://localhost:5173)..."; \
		cd opik && CREATE_DEMO_DATA=false ./opik.sh; \
	fi

.PHONY: stop
stop: ## Dummy target so 'make opik stop' parses
	@: