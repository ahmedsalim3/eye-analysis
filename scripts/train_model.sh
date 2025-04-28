# python3 -m src.modeling.train

python3 -m src.modeling.train \
    --network mi \
    --epochs 50 \
    --batch-size 64 \
    --val-split 0.15 \
    --lr 0.001 \
    --early-stop 7 \
    --lr-patience 5 \
    --random-state 42 \
    --lr 0.0005 \
