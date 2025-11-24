DATASETS_DIR = moot/optimize
COMMAND_FILE = commands.sh
NAME ?= SMAC
BASE_CMD = python3 experiment_runner_cluster.py --name $(NAME) --repeats 20 --budget 6 12 18 24 50 100 200 --runs_output_folder /share/tjmenzie/kgangul/results/results_$(NAME) --logging_folder /share/tjmenzie/kgangul/logging/logging_$(NAME) --output_directory /share/tjmenzie/kgangul/results/tmp/$(NAME)_tmp
generate-commands:
	@echo "#!/bin/bash" > $(COMMAND_FILE)
	@find $(DATASETS_DIR) -type f -name "*.csv" | while read dataset; do \
		echo "$(BASE_CMD) --datasets ../$$dataset;" >> $(COMMAND_FILE); \
	done
	@echo "wait" >> $(COMMAND_FILE)
	@chmod +x $(COMMAND_FILE)
	@mv $(COMMAND_FILE) experiments/$(COMMAND_FILE)
run-commands:
	@nohup ./$(COMMAND_FILE) > run.log 2>&1 &
	@echo "Commands are running in the background. Output is in run.log"


convert-commands:
	@mkdir -p jobs_$(NAME)
	@while read -r line; do \
		clean_line=$$(echo "$$line" | sed 's/--budget [0-9 ]*//;s/;//'); \
		dataset=$$(echo "$$clean_line" | sed 's/.*--datasets \(.*\.csv\).*/\1/'); \
		base=$$(basename "$$dataset" .csv); \
		for B in 6 12 18 24 50 100 200; do \
			jobfile=jobs_$(NAME)/job_$${base}_$${B}.lsf; \
			echo "#!/bin/bash -l" > $$jobfile; \
			echo "#BSUB -J $(NAME)_$${base}_$${B}" >> $$jobfile; \
			echo "#BSUB -n 1" >> $$jobfile; \
			echo "#BSUB -q short" >> $$jobfile; \
			echo "#BSUB -o out_%J.log" >> $$jobfile; \
			echo "#BSUB -e err_%J.log" >> $$jobfile; \
			echo "" >> $$jobfile; \
			echo "source /usr/local/apps/miniconda20240526/etc/profile.d/conda.sh" >> $$jobfile; \
			echo "conda activate /share/tjmenzie/kgangul/SEOptBench/smac_env" >> $$jobfile; \
			echo "" >> $$jobfile; \
			echo "cd /share/tjmenzie/kgangul/SEOptBench/experiments" >> $$jobfile; \
			echo "$$clean_line --budget $${B}" >> $$jobfile; \
		done; \
	done < experiments/commands.sh
