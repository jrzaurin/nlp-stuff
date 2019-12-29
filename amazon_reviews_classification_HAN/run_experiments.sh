# Pytorch
python run_pytorch.py --batch_size 32  --save_results
python run_pytorch.py --batch_size 64  --save_results
python run_pytorch.py --batch_size 128 --save_results
python run_pytorch.py --batch_size 256 --save_results
python run_pytorch.py --batch_size 512 --save_results

python run_pytorch.py --batch_size 128 --embed_dim 50  --weight_decay 0.05 --save_results
python run_pytorch.py --batch_size 128 --embed_dim 100 --weight_decay 0.05 --save_results
python run_pytorch.py --batch_size 128 --embed_dim 200 --weight_decay 0.05 --save_results
python run_pytorch.py --batch_size 128 --embed_dim 300 --weight_decay 0.05 --save_results

python run_pytorch.py --batch_size 128 --embed_dim 50 --word_hidden_dim 16 --sent_hidden_dim 16 --save_results
python run_pytorch.py --batch_size 128 --embed_dim 50 --word_hidden_dim 32 --sent_hidden_dim 32 --save_results
python run_pytorch.py --batch_size 128 --embed_dim 50 --word_hidden_dim 64 --sent_hidden_dim 64 --save_results
python run_pytorch.py --batch_size 128 --embed_dim 50 --word_hidden_dim 128 --sent_hidden_dim 128 --save_results

python run_pytorch.py --batch_size 64 --embed_dim 300 --word_hidden_dim 64 --sent_hidden_dim 64 \
--embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 --save_results
python run_pytorch.py --batch_size 64 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--embed_drop 0.5 --weight_drop 0.5 --locked_drop 0.5 --last_drop 0.5 --save_results
python run_pytorch.py --batch_size 128 --embed_dim 300 --word_hidden_dim 64 --sent_hidden_dim 64 \
--embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 --save_results
python run_pytorch.py --batch_size 128 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--embed_drop 0.5 --weight_drop 0.5 --locked_drop 0.5 --last_drop 0.5 --save_results
python run_pytorch.py --batch_size 512 --embed_dim 300 --word_hidden_dim 64 --sent_hidden_dim 64 \
--embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 --save_results
python run_pytorch.py --batch_size 512 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--embed_drop 0.5 --weight_drop 0.5 --locked_drop 0.5 --last_drop 0.5 --save_results

python run_pytorch.py --batch_size 64 --embed_dim 300 --word_hidden_dim 64 --sent_hidden_dim 64 \
--embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler reducelronplateau --lr_patience 2 --patience 4 --save_results
python run_pytorch.py --batch_size 64 --embed_dim 300 --word_hidden_dim 64 --sent_hidden_dim 64 \
--embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler cycliclr --n_cycles 1 --lr 0.0005 --patience 4 --save_results
python run_pytorch.py --batch_size 64 --embed_dim 300 --word_hidden_dim 64 --sent_hidden_dim 64 \
--embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler cycliclr --n_cycles 2 --lr 0.0005 --patience 4 --save_results
python run_pytorch.py --batch_size 64 --embed_dim 300 --word_hidden_dim 64 --sent_hidden_dim 64 \
--embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler cycliclr --n_cycles 4 --lr 0.0005 --patience 4 --save_results
python run_pytorch.py --batch_size 512 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--weight_decay 0.05 --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler reducelronplateau --lr_patience 2 --patience 4  --save_results
python run_pytorch.py --batch_size 512 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--weight_decay 0.05 --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler cycliclr --n_cycles 1 --lr 0.0005 --patience 4 --save_results
python run_pytorch.py --batch_size 512 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--weight_decay 0.05 --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler cycliclr --n_cycles 2 --lr 0.0005 --patience 4 --save_results
python run_pytorch.py --batch_size 512 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--weight_decay 0.05 --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler cycliclr --n_cycles 4 --lr 0.0005 --patience 4 --save_results

python run_pytorch.py --batch_size 128 --model rnn --save_results
python run_pytorch.py --batch_size 128 --model rnn --num_layers 3 --save_results
python run_pytorch.py --batch_size 128 --model rnn --num_layers 1 --save_results
python run_pytorch.py --batch_size 128 --model rnn --with_attention --save_results
python run_pytorch.py --batch_size 128 --model rnn --with_attention --num_layers 3 --save_results
python run_pytorch.py --batch_size 128 --model rnn --with_attention --num_layers 1 --save_results

python run_pytorch.py --batch_size 128 --embed_dim 100 --hidden_dim 64 --model rnn --save_results
python run_pytorch.py --batch_size 128 --embed_dim 100 --hidden_dim 128 --model rnn --save_results
python run_pytorch.py --batch_size 128 --embed_dim 300 --hidden_dim 64 --model rnn --save_results
python run_pytorch.py --batch_size 128 --embed_dim 300 --hidden_dim 128 --model rnn --save_results

python run_pytorch.py --batch_size 512 --embed_dim 300 --hidden_dim 128 --embed_drop 0.2 --rnn_dropout 0.2 --locked_drop 0.2 \
--last_drop 0.2 --model rnn --save_results
python run_pytorch.py --batch_size 512 --embed_dim 300 --hidden_dim 128 --weight_decay 0.05 --embed_drop 0.2 --rnn_dropout 0.2 \
--locked_drop 0.2 --last_drop 0.2 --lr_scheduler reducelronplateau --lr_patience 2 --patience 4  --model rnn --save_results
python run_pytorch.py --batch_size 512 --embed_dim 300 --hidden_dim 128 --weight_decay 0.05 --embed_drop 0.2 --rnn_dropout 0.2 \
--locked_drop 0.2 --last_drop 0.2 --lr_scheduler cycliclr --n_cycles 4 --lr 0.0005 --patience 4 --model rnn --save_results

# MXNET
python run_mxnet.py --batch_size 32  --lr 0.01 --weight_decay 0. --save_results
python run_mxnet.py --batch_size 64  --lr 0.01 --weight_decay 0. --save_results
python run_mxnet.py --batch_size 128 --lr 0.01 --weight_decay 0. --save_results
python run_mxnet.py --batch_size 256 --lr 0.01 --weight_decay 0. --save_results
python run_mxnet.py --batch_size 512 --lr 0.01 --weight_decay 0. --save_results

python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 50  --weight_decay 0.001 --save_results
python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 100 --weight_decay 0.001 --save_results
python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 200 --weight_decay 0.001 --save_results
python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 300 --weight_decay 0.001 --save_results

python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 50 --word_hidden_dim 16 --sent_hidden_dim 16 --weight_decay 0.001 --save_results
python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 50 --word_hidden_dim 32 --sent_hidden_dim 32 --weight_decay 0.001 --save_results
python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 50 --word_hidden_dim 64 --sent_hidden_dim 64 --weight_decay 0.001 --save_results
python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 50 --word_hidden_dim 128 --sent_hidden_dim 128 --weight_decay 0.001 --save_results

python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 50 --word_hidden_dim 32 --sent_hidden_dim 32 \
--weight_decay 0. --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 --save_results
python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 300 --word_hidden_dim 64 --sent_hidden_dim 64 \
--weight_decay 0. --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 --save_results
python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--weight_decay 0.001 --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 --save_results

python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 50 --word_hidden_dim 32 --sent_hidden_dim 32 \
--weight_decay 0. --lr_scheduler multifactorscheduler --steps_epochs "[2,4,8]" --save_results
python run_mxnet.py --batch_size 128 --lr 0.01 --embed_dim 200 --word_hidden_dim 64 --sent_hidden_dim 64 \
--weight_decay 0.001 --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0. \
--lr_scheduler multifactorscheduler --steps_epochs "[2,4,8]" --save_results
python run_mxnet.py --batch_size 512 --lr 0.01 --embed_dim 50 --word_hidden_dim 32 --sent_hidden_dim 32 \
--weight_decay 0.001 --lr_scheduler multifactorscheduler --steps_epochs "[2,4,8]" --save_results