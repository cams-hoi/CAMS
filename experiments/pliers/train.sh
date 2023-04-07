# sed -i 's/\r%//' filename
work_path=$(dirname $0)
cuda=${1}
nproc=${2}
port=${3}

CUDA_VISIBLE_DEVICES=${cuda} torchrun --nnodes=1 --nproc_per_node=${nproc} \
    --rdzv_endpoint=localhost:${port} \
    planner.py \
    --config=${work_path}/config.yaml
