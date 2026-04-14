DATASETS_DIR = moot/optimize
COMMAND_FILE = commands.sh
NAME ?= SMAC

# --- Separate roots for analytical test problems (DTLZ) ---
TEST_RESULTS_ROOT ?= ../results_test_problem
TEST_LOG_ROOT     ?= ../logging/logging_test_problem
TEST_TMP_ROOT     ?= ../results_test_problem/tmp_test_problem

# ---- DTLZ settings (Ignore for this repo)----
DTLZ_PROBLEMS ?= dtlz1 dtlz2 dtlz3 dtlz4 dtlz5 dtlz6 dtlz7
DTLZ_N_VARS  ?= 12
DTLZ_N_OBJS  ?= 3
DTLZ_CSV_DIR ?= data/dtlz_tables

BASE_CMD = python3 experiment_runner_cluster.py \
	--name $(NAME) --repeats 20 --budget 6 12 18 24 50 100 200 \
	--runs_output_folder ../results/results_$(NAME) \
	--logging_folder ../logging/logging_$(NAME) \
	--output_directory ../results/tmp/$(NAME)_tmp

generate-commands:
	@echo "#!/bin/bash" > $(COMMAND_FILE)
	@find $(DATASETS_DIR) -type f -name "*.csv" | while read dataset; do \
		echo "$(BASE_CMD) --datasets ../$$dataset;" >> $(COMMAND_FILE); \
	done
	@echo "wait" >> $(COMMAND_FILE)
	@chmod +x $(COMMAND_FILE)
	@mv $(COMMAND_FILE) experiments/$(COMMAND_FILE)

generate-dtlz-csv-commands:
	@echo "#!/bin/bash" > $(COMMAND_FILE)
	@find $(DTLZ_CSV_DIR) -type f -name "*.csv" | while read dataset; do \
		base=$$(basename $$dataset .csv); \
		echo "$(BASE_CMD) \
			--datasets ../$$dataset \
			--runs_output_folder $(TEST_RESULTS_ROOT)/results_$(NAME)/$$base \
			--logging_folder $(TEST_LOG_ROOT)/logging_$(NAME)/$$base \
			--output_directory $(TEST_TMP_ROOT)/$(NAME)_tmp/$$base;" \
		>> $(COMMAND_FILE); \
	done
	@echo "wait" >> $(COMMAND_FILE)
	@chmod +x $(COMMAND_FILE)
	@mv $(COMMAND_FILE) experiments/$(COMMAND_FILE)

generate-dtlz-commands:
	@echo "#!/bin/bash" > $(COMMAND_FILE)
	@for p in $(DTLZ_PROBLEMS); do \
		tag=$$p"_v$(DTLZ_N_VARS)_m$(DTLZ_N_OBJS)"; \
		echo "python3 experiment_runner_cluster.py \
			--name $(NAME) \
			--repeats 20 \
			--budget 6 12 18 24 50 100 200 \
			--problem $$p \
			--n_vars $(DTLZ_N_VARS) \
			--n_objs $(DTLZ_N_OBJS) \
			--runs_output_folder $(TEST_RESULTS_ROOT)/results_$(NAME)/$$tag \
			--logging_folder $(TEST_LOG_ROOT)/logging_$(NAME)/$$tag \
			--output_directory $(TEST_TMP_ROOT)/$(NAME)_tmp/$$tag;" \
		>> $(COMMAND_FILE); \
	done
	@echo "wait" >> $(COMMAND_FILE)
	@chmod +x $(COMMAND_FILE)
	@mv $(COMMAND_FILE) experiments/$(COMMAND_FILE)

run-commands:
	@cd experiments && nohup ./$(COMMAND_FILE) > run.log 2>&1 &
	@echo "Commands are running in the background. Output is in experiments/run.log"

convert-commands:
	@mkdir -p jobs_$(NAME)
	@while read -r line; do \
		clean_line=$$(echo "$$line" | sed 's/--budget [0-9 ]*//;s/;//'); \
		dtlz=$$(echo "$$clean_line" | sed -n 's/.*--dtlz \([^ ]*\).*/\1/p'); \
		dataset=$$(echo "$$clean_line" | sed -n 's/.*--datasets \([^ ]*\.csv\).*/\1/p'); \
		if [ -n "$$dtlz" ]; then \
			nv=$$(echo "$$clean_line" | sed -n 's/.*--n_vars \([0-9]*\).*/\1/p'); \
			no=$$(echo "$$clean_line" | sed -n 's/.*--n_objs \([0-9]*\).*/\1/p'); \
			base=$$dtlz"_v"$${nv}"_m"$${no}; \
		else \
			base=$$(basename "$$dataset" .csv); \
		fi; \
		for B in 6 12 18 24 50 100 200; do \
			jobfile=jobs_$(NAME)/job_$${base}_$${B}.lsf; \
			echo "#!/bin/bash -l" > $$jobfile; \
			echo "#BSUB -J $(NAME)_$${base}_$${B}" >> $$jobfile; \
			echo "#BSUB -n 1" >> $$jobfile; \
			echo "#BSUB -q short" >> $$jobfile; \
			echo "#BSUB -W 120" >> $$jobfile; \
			echo "#BSUB -o out_%J.log" >> $$jobfile; \
			echo "#BSUB -e err_%J.log" >> $$jobfile; \
			echo "" >> $$jobfile; \
			echo "source /usr/local/apps/miniconda20240526/etc/profile.d/conda.sh" >> $$jobfile; \
			echo "conda activate /share/NEOuser/NEO/env" >> $$jobfile; \
			echo "" >> $$jobfile; \
			echo "cd /share/NEOuser/NEO/experiments" >> $$jobfile; \
			echo "$$clean_line --budget $${B}" >> $$jobfile; \
		done; \
	done < experiments/$(COMMAND_FILE)

submit-jobs:
	@for f in jobs_$(NAME)/*.lsf; do \
		echo "bsub < $$f"; \
		bsub < $$f; \
	done
