# BiaffineDParser
===========================    
A re-implementation of "[Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)" in Pytorch
    
Pytorch version 0.3.0   
Python: 3.6   

## Run:
python3.6 driver/TrainTest.py --config_file config.ctb51.cfg

## Results:
CTB51:  90.84 (UAS) 89.57 (LAS)    
PTB (Stanford Dependencies) :  95.84 (UAS) 94.15 (LAS)    
