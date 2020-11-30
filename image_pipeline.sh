# Defining the dataset to be used
DATA="cifar10"

# Architecture
MODEL="mlp"

# Number of input units
N_INPUT=3072

# Number of hidden units
N_HIDDEN=128

# Number of classes
N_CLASS=10

# Learning rate
LR=0.001

# Batch size
BATCH_SIZE=100

# Training epochs
EPOCHS=1

# Layer to be optimized
OPT_LAYER="fc2"

# Meta-heuristic
MH="pso"

# Optimization bounds 
BOUNDS=0.01

# Number of agents
N_AGENTS=1

# Number of iterations
N_ITER=1

# Device
DEVICE="cpu"

# Defining the number of runnings
N_RUNS=1

# Iterates through all possible seeds
for SEED in $(seq 1 $N_RUNS); do
    # Trains an architecture
    python image_model_training.py ${DATA} ${MODEL} ${MODEL}_${DATA}_${SEED}.pth -n_input ${N_INPUT} -n_hidden ${N_HIDDEN} -n_class ${N_CLASS} -lr ${LR} -batch_size ${BATCH_SIZE} -epochs ${EPOCHS} -device ${DEVICE} -seed ${SEED}

    # Optimizes the architecture
    python image_model_optimization.py ${DATA} ${MODEL}_${DATA}_${SEED}.pth ${OPT_LAYER} ${MH} -batch_size ${BATCH_SIZE} -bounds ${BOUNDS} -n_agents ${N_AGENTS} -n_iter ${N_ITER} -seed ${SEED}

    # Evaluates the architecture
    python image_model_evaluation.py ${DATA} ${MODEL}_${DATA}_${SEED}.pth_${MH}.optimized -batch_size ${BATCH_SIZE} -seed ${SEED}
done