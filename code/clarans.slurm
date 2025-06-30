#!/bin/bash
#SBATCH -J GOS_clarans
#SBATCH -o /nfsd/bcb/bcbg/meneghinma/results/output.txt
#SBATCH -e /nfsd/bcb/bcbg/meneghinma/results/errors.txt
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH --mem 250G
#SBATCH --mail-user matteo.meneghin@studenti.unipd.it
#SBATCH --mail-type ALL

source /nfsd/bcb/bcbg/meneghinma/scripts/.venv/bin/activate

srun python /nfsd/bcb/bcbg/meneghinma/scripts/clarans.py