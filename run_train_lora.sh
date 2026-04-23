#!/bin/bash
#SBATCH --job-name=medgemma_sft_lora
#SBATCH --account=PAS3128
#SBATCH --time=120:00:00
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=32      
#SBATCH --gpus-per-node=2       
#SBATCH --partition=nextgen
#SBATCH --output=medgemma_sft_lora-%j.out 

module load miniconda3/24.1.2-py310 cuda/12.8.1
source activate /fs/ess/PAS3128/ruoxiyang/env/medgemma_env

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_IP=$(srun --nodes=1 --ntasks=1 -w $MASTER_ADDR hostname -I | awk '{print $1}')

export NCCL_SOCKET_IFNAME=em3
export GLOO_SOCKET_IFNAME=em3
export NCCL_IB_DISABLE=1         
export NCCL_P2P_DISABLE=1        
export NCCL_DEBUG=INFO           
export TORCH_DISTRIBUTED_DEBUG=DETAIL 

export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600         

srun python -m torch.distributed.run \
    --nnodes=16 \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP:25678 \
    /fs/ess/PAS3128/ruoxiyang/code/train_lora.py