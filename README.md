
# Discriminability objective for training descriptive captions
This code is an adaptation of Luo et al.'s DiscCaptioning [repository](https://github.com/ruotianluo/DiscCaptioning), updated for Python 3. 

This is the implementation of thepaper [**Discriminability objective for training descriptive captions**](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Discriminability_Objective_for_CVPR_2018_paper.pdf).


## Requirements
Python 3.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for python 3)

PyTorch 1.8.0 (along with torchvision)

## Downloads

### Clone the repository

### Data split

In this paper we train on the COCO 2014 training set and evaluate on the COCO 2014 validation set. 

Download link: [Google drive link](https://drive.google.com/open?id=1Z9bfvkRT5YyikmNgzPbybezYj9mi4TE2)

### coco-caption

```bash
cd coco-caption
bash ./get_stanford_models.sh
cd annotations
# Download captions_val2014.json from the google drive link above to this folder
cd ../../

```

The reason why we need to replace the `captions_val2014.json` is because the original file can only evaluate images from the val2014 set, and we are using rama's split.

### Pre-computed feature

In this paper, for retrieval model, we use outputs of last layer of resnet-101. For captioning model, we use the bottom-up feature from [https://arxiv.org/abs/1707.07998](https://arxiv.org/abs/1707.07998).

The features can be downloaded from the same link, and you need to compress them to `data/cocotalk_fc` and `data/cocobu_att` respectively.

## Pretrained models.

Download pretrained models from [link](https://drive.google.com/open?id=1_-OpcVmiZ8D4OJH76D0J1l8WXZH-HQmR). Decompress them into root folder.

To evaluate on pretrained model, run:

`bash eval.sh att_d1 test`

The pretrained models can match the results shown in the paper.

## Train on your own

### Preprocessing 
Preprocess the captions (skip if you already have 'cocotalk.json' and 'cocotalk_label.h5'):
```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```
Preprocess for self-critical training:

```
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

### Start training

First train a retrieval model:

`bash run_fc_con.sh`

Second, pretrain the captioning model.

`bash run_att.sh`

Third, finetune the  captioning model with cider+discriminability optimization:

`bash run_att_d.sh 1` (1 is the discriminability weight, and can be changed to other values)

### Evaluate

```bash
bash eval.sh att_d1 test
```

## Acknowledgements

The code is based on [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch)
