node_rank=$1
export CUDA_VISIBLE_DEVICES="6,7"
INIT_FILE=COMMON_PATH/ddp_init
init_method=file://$(readlink -f $INIT_FILE)
num_nodes=2
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
world_size=`expr $num_gpus \* $num_nodes`

for ((i = 0; i < $num_gpus; ++i)); do
{
rank=`expr $node_rank \* $num_gpus + $i`
python train_file.py --world_size=$world_size --local_rank=$rank --file_method=$init_method
} &
done
