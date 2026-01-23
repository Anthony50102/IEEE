#!/bin/bash
#=============================================================================
# FNO Pipeline - SLURM Launcher for TACC Vista
#
# Usage:
#   sbatch run_fno.slurm [step] [config] [run_dir]
#
# Examples:
#   sbatch run_fno.slurm 1 config/fno_temporal_split.yaml
#   sbatch run_fno.slurm 2 config/fno_temporal_split.yaml /path/to/run_dir
#   sbatch run_fno.slurm 3 config/fno_temporal_split.yaml /path/to/run_dir
#   sbatch run_fno.slurm all config/fno_temporal_split.yaml
#
# Steps:
#   1 - Single-step FNO training
#   2 - Autoregressive rollout training
#   3 - Evaluation and prediction
#   all - Run all steps sequentially
#
#=============================================================================

#SBATCH -J fno_train
#SBATCH -o fno_%j.out
#SBATCH -e fno_%j.err
#SBATCH -p gh              # GPU partition (adjust for your system)
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00         # 24 hours for long runs

set -e

# =============================================================================
# Configuration
# =============================================================================

STEP=${1:-1}
CONFIG=${2:-fno/config/fno_temporal_split.yaml}
RUN_DIR=${3:-}

# =============================================================================
# Environment Setup
# =============================================================================

# Source startup script to configure virtual environment
# Modify this path to your actual startup script location
if [ -f "$WORK/single_node_startup.sh" ]; then
    source $WORK/single_node_startup.sh
elif [ -f "$HOME/single_node_startup.sh" ]; then
    source $HOME/single_node_startup.sh
elif [ -f "/home1/10407/anthony50102/single_node_startup.sh" ]; then
    source /home1/10407/anthony50102/single_node_startup.sh
else
    # Fallback: load modules directly
    echo "No startup script found"
    exit 1
fi

# Navigate to project directory
cd $WORK/repos/IEEE

# =============================================================================
# Job Info
# =============================================================================

echo "=============================================="
echo "FNO Training Pipeline"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Step: $STEP"
echo "Config: $CONFIG"
echo "Run dir: ${RUN_DIR:-'(auto-generated)'}"
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "=============================================="

# Print GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "GPU info unavailable"
echo "=============================================="

# =============================================================================
# Run Pipeline
# =============================================================================

case $STEP in
    1)
        echo "Running Step 1: Single-step FNO training..."
        python fno/step_1_train.py --config $CONFIG --test
        ;;
    
    2)
        if [ -z "$RUN_DIR" ]; then
            echo "ERROR: Step 2 requires --run-dir from Step 1"
            echo "Usage: sbatch run_fno.slurm 2 config.yaml /path/to/run_dir"
            exit 1
        fi
        echo "Running Step 2: Autoregressive rollout training..."
        python fno/step_2_train.py --config $CONFIG --run-dir $RUN_DIR --test
        ;;
    
    3)
        if [ -z "$RUN_DIR" ]; then
            echo "ERROR: Step 3 requires --run-dir from Steps 1/2"
            echo "Usage: sbatch run_fno.slurm 3 config.yaml /path/to/run_dir"
            exit 1
        fi
        echo "Running Step 3: Evaluation and prediction..."
        python fno/step_3_evaluate.py --config $CONFIG --run-dir $RUN_DIR
        ;;
    
    all)
        echo "Running Full Pipeline (Steps 1 + 2 + 3)..."
        echo ""
        
        # Step 1
        echo "=== Step 1: Single-step training ==="
        python fno/step_1_train.py --config $CONFIG --test
        
        # Find the most recent run directory
        OUTPUT_BASE=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg.get('paths',{}).get('output_base','./output/'))")
        RUN_NAME=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg.get('run_name','fno_experiment'))")
        LATEST_RUN=$(ls -td ${OUTPUT_BASE}/${RUN_NAME}_* 2>/dev/null | head -1)
        
        if [ -z "$LATEST_RUN" ]; then
            echo "ERROR: Could not find run directory from Step 1"
            exit 1
        fi
        
        echo "Found run directory: $LATEST_RUN"
        echo ""
        
        # Step 2
        echo "=== Step 2: Autoregressive rollout training ==="
        python fno/step_2_train.py --config $CONFIG --run-dir $LATEST_RUN --test
        
        # Step 3
        echo ""
        echo "=== Step 3: Evaluation and prediction ==="
        python fno/step_3_evaluate.py --config $CONFIG --run-dir $LATEST_RUN
        ;;
    
    *)
        echo "ERROR: Unknown step '$STEP'"
        echo "Valid options: 1, 2, 3, all"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
