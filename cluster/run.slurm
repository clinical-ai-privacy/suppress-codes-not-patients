#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Generic runner, submitted from the code directory (mkdir logs first):
#   sbatch -J <name> cluster/run.slurm scripts/<script>.py [args]
# Adjust the partition, the container image (SIF) and KM_HOME / KM_DATA to your cluster.
set -u
umask 077
export KM_HOME=${KM_HOME:-$HOME/km} KM_DATA=${KM_DATA:-$HOME/km_data}
export KM_WORKERS=${SLURM_CPUS_PER_TASK:-8} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SIF=${SIF:-$HOME/kmrel.sif}
echo "=== $(date '+%F %T') node=$(hostname) $*"
if [ -f "$SIF" ]; then
  apptainer exec --bind "$KM_HOME,$KM_DATA" "$SIF" python "$@"
else
  python3 "$@"
fi
echo "=== rc=$? $(date '+%F %T')"
