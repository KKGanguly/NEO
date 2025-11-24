DATASETS_DIR = moot/optimize
COMMAND_FILE = commands.sh
NAME ?= SMAC
BASE_CMD = python3 experiment_runner_cluster.py --name $(NAME) --repeats 20 --budget 6 12 18 24 50 100 200 --runs_output_folder ../results/results_$(NAME) --logging_folder ../logging/logging_$(NAME) --output_directory ../results/tmp/$(NAME)_tmp
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
	@mkdir -p jobs_smac
	@while read -r line; do \
		dataset=$$(echo "$$line" | sed 's/.*--datasets \(.*\.csv\).*/\1/'); \
		base=$$(basename "$$dataset" .csv); \
		for B in 6 12 18 24 50 100 200; do \
			jobfile=jobs_smac/job_$${base}_$${B}.lsf; \
			echo "#!/bin/bash" > $$jobfile; \
			echo "#BSUB -J SMAC_$${base}_$${B}" >> $$jobfile; \
			echo "#BSUB -n 1" >> $$jobfile; \
			echo "#BSUB -W 02:00" >> $$jobfile; \
			echo "#BSUB -q short" >> $$jobfile; \
			echo "#BSUB -o out_%J.log" >> $$jobfile; \
			echo "#BSUB -e err_%J.log" >> $$jobfile; \
			echo "" >> $$jobfile; \
			echo "module load anaconda" >> $$jobfile; \
			echo "" >> $$jobfile; \
			echo "# Create environment if missing" >> $$jobfile; \
			echo "conda env list | grep smac_env >/dev/null 2>&1 || conda create -n smac_env python=3.10 -y" >> $$jobfile; \
			echo "source activate smac_env" >> $$jobfile; \
			echo "" >> $$jobfile; \
			echo "# Install required Python packages" >> $$jobfile; \
			echo "python -m pip install -r requirements.txt" >> $$jobfile; \
			echo "" >> $$jobfile; \
			echo "# Run the actual job" >> $$jobfile; \
			echo "$$line --budget $${B}" >> $$jobfile; \
		done; \
	done < experiments/commands.sh

