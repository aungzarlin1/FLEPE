# #!/bin/bash

# for seed in {0..9}
# do
#   python3 main_directed_node_classification.py \
#     --dataset cora_ml \
#     --model gcn-flepe \
#     --flow_method PRE \
#     --vertex_import True \
#     --k 10 \
#     --seed $seed
# done

for k in {5..20}
do
  for seed in {0..9}
  do
    python3 main_directed_node_classification.py \
      --dataset telegram \
      --model gcn-flepe \
      --flow_method RGE \
      --vertex_import False \
      --k $k \
      --seed $seed
  done
done
