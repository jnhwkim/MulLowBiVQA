# Hadamard Product for Low-rank Bilinear Pooling

Multimodal Low-rank Bilinear Attention Networks (MLB) have an efficient attention mechanism by low-rank bilinear pooling for visual question-answering tasks. MLB achieves a new state-of-the-art performance, having a better parsimonious property than previous methods.

This current code can get **65.07** on Open-Ended and **68.89** on Multiple-Choice on **test-standard** split for the [VQA dataset](http://visualqa.org). For an ensemble model, **66.89** and **70.29**, resepectively.

### Dependencies

* [rnn](https://github.com/Element-Research/rnn)

You can install the dependencies:

```bash
luarocks install rnn
```

### Training

Please follow the instruction from [VQA_LSTM_CNN](https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/readme.md) for preprocessing. `--split 2` option allows to use train+val set to train, and test-dev or test-standard set to evaluate. Set `--num_ans` to `2000` to reproduce the result.

For question features, you need to use this:

* [skip-thoughts](https://github.com/ryankiros/skip-thoughts)
* [DPPnet](https://github.com/HyeonwooNoh/DPPnet) (see 003_skipthoughts_porting)
* `make_lookuptable.lua`

for image features,

```
$ th prepro_res.lua -input_json data_train-val_test-dev_2k/data_prepro.json -image_root path_to_image_root -cnn_model path to cnn_model
```

The pretrained ResNet-152 model and related scripts can be found in [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua).

```
$ th train.lua
``` 

With the default parameter, this will take around 2.6 days on a sinlge NVIDIA Titan X GPU, and will generate the model under `model/`. **For the result of the paper, use `-seconds` option for `answer sampling` in Section 5. `seconds.json` file can be optained using `prepro_seconds.lua`.**

### Evaluation

```
$ th eval.lua
```

### References

If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:

```
@inproceedings{Kim2016c,
author = {Kim, Jin-Hwa and On, Kyoung-Woon and Kim, Jeonghee and Ha, Jung-Woo and Zhang, Byoung-Tak},
booktitle = {5th International Conference on Learning Representations},
title = {{Hadamard Product for Low-rank Bilinear Pooling}},
archivePrefix = {arXiv},
arxivId = {1610.04325},
year = {2017}
}
```

This code uses Torch7 `rnn` package and its `TrimZero` module for question embeddings. Notice that following papers:

```
@article{Leonard2015a,
author = {L{\'{e}}onard, Nicholas and Waghmare, Sagar and Wang, Yang and Kim, Jin-Hwa},
journal = {arXiv preprint arXiv:1511.07889},
title = {{rnn : Recurrent Library for Torch}},
year = {2015}
}
@inproceedings{Kim2016a,
author = {Kim, Jin-Hwa and Kim, Jeonghee and Ha, Jung-Woo and Zhang, Byoung-Tak},
booktitle = {Proceedings of KIIS Spring Conference},
isbn = {2093-4025},
number = {1},
pages = {165--166},
title = {{TrimZero: A Torch Recurrent Module for Efficient Natural Language Processing}},
volume = {26},
year = {2016}
}
```

### License

BSD 3-Clause License
  
### Patent (Pending)

METHOD AND SYSTEM FOR PROCESSING DATA USING ELEMENT-WISE MULTIPLICATION AND MULTIMODAL RESIDUAL LEARNING FOR VISUAL QUESTION-ANSWERING
