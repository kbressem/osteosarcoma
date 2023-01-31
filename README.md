# Deep learning segmentation and classification of bone tumors in children

## Getting started: 

This project runs on [trainlib v0.4](https://github.com/kbressem/trainlib/tree/v0.4). 

```
git clone https://github.com/kbressem/trainlib/tree/v0.2
cd trainlib
pip install -e . 
```
**Note** It might be beneficial to install PyTorch with conda instead of pip, if you run into problems with CUDA version. 

For image registration, `SimpleITK` and `SimpleElastix` are needed. This project uses Version `2.0.0rc2.dev908-g8244e`.

`pip install SimpleITK-SimpleElastix`

