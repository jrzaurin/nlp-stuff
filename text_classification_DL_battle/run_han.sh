python run_han.py --batch_size 512 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--weight_decay 0.05 --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler cycliclr --n_cycles 4 --lr 0.0005 --patience 4 --save_results

python run_han.py --batch_size 512 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--weight_decay 0. --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler reducelronplateau --lr_patience 2 --patience 4  --save_results

python run_han.py --batch_size 512 --embed_dim 300 --word_hidden_dim 128 --sent_hidden_dim 128 \
--weight_decay 0.05 --embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler reducelronplateau --lr_patience 2 --patience 4  --save_results
