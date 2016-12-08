# Hadamard Product for Low-rank Bilinear Pooling

Multimodal Low-rank Bilinear Attention Networks (MLB) have an efficient attention mechanism by low-rank bilinear pooling for visual question-answering tasks. MLB achieves a new state-of-the-art performance, having a better parsimonious property than previous methods.

This current code can get **65.07** on Open-Ended and **68.89** on Multiple-Choice on **test-standard** split for the [VQA dataset](http://visualqa.org). For an ensemble model, **66.89** and **70.29**, resepectively.

For now, the model definition is available. We're polishing messy codes and confirming the whole steps to reproduce paper results seamlessly. Stay tuned for upcoming updates!

### References

If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:

```
@article{Kim2016c,
author = {Kim, Jin-Hwa and On, Kyoung-Woon and Kim, Jeonghee and Ha, Jung-Woo and Zhang, Byoung-Tak},
title = {{Hadamard Product for Low-rank Bilinear Pooling}},
url = {http://arxiv.org/abs/1610.04325},
year = {2016}
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
