# NeuralRST-TopDown

## Paper
Fajri Koto, Jey Han Lau, and Timothy Baldwin. [_Top-down Discourse Parsing via Sequence Labelling_](https://www.aclweb.org/anthology/2021.eacl-main.60.pdf). 
In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2021): Main Volume. 

## About the code
This code uses LSTM, if you want to use the Transformer version please find it [here](https://github.com/fajri91/NeuralRST-TopDown-Transformer). 
The encoder is designed based on [Yu et al., 2018](https://github.com/yunan4nlp/NNDisParser) where we use three main embeddings: 
1) Word embedding, initialized by [glove.6B.200d.txt.gz](https://nlp.stanford.edu/projects/glove/).
2) POS Tags embedding, initialized randomly.
3) Syntax Embedding from [BiAffine Dependency Parser](https://arxiv.org/abs/1611.01734). Please refer to [RSTExtractor](https://github.com/fajri91/RSTExtractor) to see how we extract it

## Dependencies 
1. Python 3.6
2. Run `pip install -r requirements.txt`

## Data and Resource

We use [English RST Tree Bank](https://catalog.ldc.upenn.edu/LDC2002T07). Please make sure you have a right 
to access this data. Our code uses the input of the binarized discourse tree, provided by Yu et al., 2018. 

In this repository, we do not provide you with the raw RST Tree Bank, but the binarized version split 
in train/dev/test based on [Yu et al., 2018](https://github.com/yunan4nlp/NNDisParser). We also provide
the extracted syntax feature for each data split. Please download them [here](https://drive.google.com/file/d/1mSS6Nj8vkiU9Q6q8r-I7p2fh44NOdFsZ/view).

## Running the code

In the experiment we use 1 GPU V100 (16GB).

For training the LSTM with static oracle (normal training)
```
CUDA_VISIBLE_DEVICES=0 python train_rst_parser.py --experiment=exp_static \
                        --word_embedding_file=[path_to_glove] \
                        --train=[path_to_train_data] --test=[path_to_test_data] --dev=[path_to_dev_data] \
                        --train_syn_feat=[path_to_syntax_feature_of_train] \
                        --test_syn_feat=[path_to_syntax_feature_of_test] \
                        --dev_syn_feat=[path_to_syntax_feature_of_dev] \
                        --max_sent_size=100 --hidden_size=256 --hidden_size_tagger=128 --batch_size=4 \
                        --grad_accum=2 --lr=0.001 --ada_eps=1e-6 --gamma=1e-6 \
                        --loss_seg=1.0 --loss_nuc_rel=1.0 --depth_alpha=0 \
                        --elem_alpha=0.35
```

For training the LSTM with the dynamic oracle:
```
CUDA_VISIBLE_DEVICES=0 python train_rst_parser.py --experiment=exp_dynamic \
                        --word_embedding_file=[path_to_glove] \
                        --train=[path_to_train_data] --test=[path_to_test_data] --dev=[path_to_dev_data] \
                        --train_syn_feat=[path_to_syntax_feature_of_train] \
                        --test_syn_feat=[path_to_syntax_feature_of_test] \
                        --dev_syn_feat=[path_to_syntax_feature_of_dev] \
                        --max_sent_size=100 --hidden_size=256 --hidden_size_tagger=128 --batch_size=4 \
                        --grad_accum=2 --lr=0.001 --ada_eps=1e-6 --gamma=1e-6 \
                        --loss_seg=1.0 --loss_nuc=1.0 --beam_search=1 --depth_alpha=0 \
                        --elem_alpha=0.35 --use_dynamic_oracle=1 --start_dynamic_oracle=50 --oracle_prob=0.65
```

## Models

We also provide the result of static and dynamic training of our LSTM model. Please download them [here](https://drive.google.com/file/d/1u1uEN1BfpMIJX47iZIxAk3ojzHZHxxxO/view?usp=sharing).
To run this model, please download all data for the input, and adjust the `config.cfg` accordingly. This includes `word_embedding_file`, `train_path`, `test_path`,
`dev_path`, `train_syn_feat_path`, `dev_syn_feat_path`, `test_syn_feat_path`, `model_path`, `model_name`, `alphabet_path`.

You can run the model by
```
CUDA_VISIBLE_DEVICES=0 python run_rst_parser.py --config_path=path_to_config.cfg
```

Output for static training, (rst = RST Parseval, ori = original Parseval):
```
Reading dev instance, and predict...
S (rst): Recall: R=3338/3886=0.859, Precision: P=3338/3886=0.859, Fmeasure: 0.859
N (rst): Recall: R=2840/3886=0.7308, Precision: P=2840/3886=0.7308, Fmeasure: 0.7308
R (rst): Recall: R=2389/3886=0.6148, Precision: P=2389/3886=0.6148, Fmeasure: 0.6148
F (rst): Recall: R=2372/3886=0.6104, Precision: P=2372/3886=0.6104, Fmeasure: 0.6104
-----------------------------------------------------------------------------------
S (ori): Recall: R=1395/1943=0.718, Precision: P=1395/1943=0.718, Fmeasure: 0.718
N (ori): Recall: R=1203/1943=0.6191, Precision: P=1203/1943=0.6191, Fmeasure: 0.6191
R (ori): Recall: R=1022/1943=0.526, Precision: P=1022/1943=0.526, Fmeasure: 0.526
F (ori): Recall: R=1004/1943=0.5167, Precision: P=1004/1943=0.5167, Fmeasure: 0.5167

Reading test instance, and predict...
S (rst): Recall: R=3998/4616=0.8661, Precision: P=3998/4616=0.8661, Fmeasure: 0.8661
N (rst): Recall: R=3419/4616=0.7407, Precision: P=3419/4616=0.7407, Fmeasure: 0.7407
R (rst): Recall: R=2827/4616=0.6124, Precision: P=2827/4616=0.6124, Fmeasure: 0.6124
F (rst): Recall: R=2811/4616=0.609, Precision: P=2811/4616=0.609, Fmeasure: 0.609
-----------------------------------------------------------------------------------
S (ori): Recall: R=1690/2308=0.7322, Precision: P=1690/2308=0.7322, Fmeasure: 0.7322
N (ori): Recall: R=1440/2308=0.6239, Precision: P=1440/2308=0.6239, Fmeasure: 0.6239
R (ori): Recall: R=1167/2308=0.5056, Precision: P=1167/2308=0.5056, Fmeasure: 0.5056
F (ori): Recall: R=1148/2308=0.4974, Precision: P=1148/2308=0.4974, Fmeasure: 0.4974
```

Output for dynamic training, (rst = RST Parseval, ori = original Parseval):
```
Reading dev instance, and predict...
S (rst): Recall: R=3349/3886=0.8618, Precision: P=3349/3886=0.8618, Fmeasure: 0.8618
N (rst): Recall: R=2858/3886=0.7355, Precision: P=2858/3886=0.7355, Fmeasure: 0.7355
R (rst): Recall: R=2422/3886=0.6233, Precision: P=2422/3886=0.6233, Fmeasure: 0.6233
F (rst): Recall: R=2409/3886=0.6199, Precision: P=2409/3886=0.6199, Fmeasure: 0.6199
-----------------------------------------------------------------------------------
S (ori): Recall: R=1406/1943=0.7236, Precision: P=1406/1943=0.7236, Fmeasure: 0.7236
N (ori): Recall: R=1228/1943=0.632, Precision: P=1228/1943=0.632, Fmeasure: 0.632
R (ori): Recall: R=1035/1943=0.5327, Precision: P=1035/1943=0.5327, Fmeasure: 0.5327
F (ori): Recall: R=1019/1943=0.5244, Precision: P=1019/1943=0.5244, Fmeasure: 0.5244

Reading test instance, and predict...
S (rst): Recall: R=4005/4616=0.8676, Precision: P=4005/4616=0.8676, Fmeasure: 0.8676
N (rst): Recall: R=3399/4616=0.7364, Precision: P=3399/4616=0.7364, Fmeasure: 0.7364
R (rst): Recall: R=2849/4616=0.6172, Precision: P=2849/4616=0.6172, Fmeasure: 0.6172
F (rst): Recall: R=2818/4616=0.6105, Precision: P=2818/4616=0.6105, Fmeasure: 0.6105
-----------------------------------------------------------------------------------
S (ori): Recall: R=1697/2308=0.7353, Precision: P=1697/2308=0.7353, Fmeasure: 0.7353
N (ori): Recall: R=1440/2308=0.6239, Precision: P=1440/2308=0.6239, Fmeasure: 0.6239
R (ori): Recall: R=1206/2308=0.5225, Precision: P=1206/2308=0.5225, Fmeasure: 0.5225
F (ori): Recall: R=1173/2308=0.5082, Precision: P=1173/2308=0.5082, Fmeasure: 0.5082
```

