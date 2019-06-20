SAVE=$1
export CUDA_VISIBLE_DEVICES=1

python3.6 ./train.py ../rotowire_gen \
        --arch select_plan \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 0.001 --min-lr 1e-09 \
        --weight-decay 0\
        --batch-size 5 --dropout 0.3 --save-dir $SAVE\
        --update-freq 20 --log-interval 50 \
        --num-workers 1 --max-epoch 50 \
	--source-features value key type state --task data_to_text \
	--target-lang plan --source-lang value \
	--encoder-embed-dim 600 --encoder-ffn-embed-dim 600 \
	--decoder-ffn-embed-dim 600 --lazy-load \
	--seed 1288
#        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 100
