# Defining the dataset to be used
DATA="imdb"

# Architecture
MODEL="lstm"

# Number of embedding units
N_EMBEDDING=256

# Number of hidden units
N_HIDDEN=512

# Number of classes
N_CLASS=2

# Learning rate
LR=0.001

# Batch size
BATCH_SIZE=100

# Training epochs
EPOCHS=10

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

# Defining the seed
SEED=0

# Trains an architecture
python text_model_training.py ${DATA} ${MODEL} ${MODEL}_${SEED}.pth -n_embedding ${N_INPUT} -n_hidden ${N_HIDDEN} -n_class ${N_CLASS} -lr ${LR} -batch_size ${BATCH_SIZE} -epochs ${EPOCHS} -device ${DEVICE} -seed ${SEED}

# Optimizes the architecture
python text_model_optimization.py ${DATA} ${MODEL}_${SEED}.pth ${OPT_LAYER} ${MH} -batch_size ${BATCH_SIZE} -bounds ${BOUNDS} -n_agents ${N_AGENTS} -n_iter ${N_ITER} -seed ${SEED}
