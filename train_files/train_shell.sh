# 启动screen 
screen -S train_01

# 开始训练
bash run_gpu0.sh train_dn169v1.py

# 退出screen
ctrl + a + d

# 启动tensorboard（先启动一个screen）
python tensorboard --logdir=models_and_logs --ip 0.0.0.0
