## Masked Autoencoders (MAE)

This is my personal copy of Facebook's [Masked Autoencoders (MAE)](https://github.com/facebookresearch/mae) repository for image-based SSL customized for my own purposes. The code here can be used to train and evaluate MAEs.

### Usage examples

* **Training:** To train an MAE model with a ViT-S/16 architecture from scratch on your data, use [`train_mae.py`](https://github.com/eminorhan/mae/blob/master/train_mae.py): 
```python
python -u train_mae.py \
	--model 'mae_vit_small_patch16' \
	--batch_size_per_gpu 512 \
	--num_workers 16 \
	--lr 0.0003 \
	--min_lr 0.0003 \
	--output_dir OUTPUT_DIR \
	--data_path DATA_PATH \
	--save_prefix INFORMATIVE_SAVE_PREFIX
```
This version uses the [`webdataset`](https://github.com/webdataset/webdataset) interface to feed the data into the model. There's a separete training file that uses the standard `torch`-`torchvision` data loading interface instead, if you'd prefer to use that: [`train_mae_nowds.py`](https://github.com/eminorhan/mae/blob/master/train_mae_nowds.py).

* **Linear evaluation:** To evaluate an MAE model with the linear probing approach, use [`eval_linear.py`](https://github.com/eminorhan/mae/blob/master/eval_linear.py): 
```python
python -u eval_linear.py \
	--model 'vit_small_patch16' \
	--resume MODEL_PATH \
	--save_prefix INFORMATIVE_SAVE_PREFIX \
	--batch_size 1024 \
	--epochs 50 \
	--num_workers 16 \
	--lr 0.0003 \
	--output_dir OUTPUT_DIR \
	--train_data_path TRAIN_DATA_PATH \
	--val_data_path VAL_DATA_PATH \
	--num_labels 1000
```

* **Finetuning evaluation:** To evaluate an MAE model with the finetuning approach, use [`eval_finetune.py`](https://github.com/eminorhan/mae/blob/master/eval_finetune.py): 
```python
python -u eval_finetune.py \
	--model 'vit_small_patch16' \
	--resume MODEL_PATH \
	--save_prefix INFORMATIVE_SAVE_PREFIX \
	--batch_size 128 \
	--epochs 50 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir OUTPUT_DIR \
	--train_data_path TRAIN_DATA_PATH \
	--val_data_path VAL_DATA_PATH \
	--frac_retained 0.010147 \
	--num_labels 1000
```
Here `frac_retained` is the fraction of the training set used for finetuning and can be set to do few-shot finetuning evals (*e.g.* `--frac_retained 0.01` corresponds to finetuning with 1% of the training data, *i.e.* 12-13 examples per class in the case of ImageNet).

### Datasets

* **SAYCAM:**              1M 224x224 blur crops     | 400 folders - 2500 each | ~30 fps |
* **SAYCAM_blur:**         200K 224x224 blur crops   | 400 folders - 500 each  | ~6 fps  |
* **SAYCAM_blur_full:**    200K 640x480 blur frames  | 400 folders - 500 each  | ~6 fps  |
* **SAYCAM_deblur:**       200K 224x224 deblur crops | 400 folders - 500 each  | ~6 fps  | Notes: deblur with downsampling first
* **SAYCAM_deblur_new:**   200K 224x224 deblur crops | 400 folders - 500 each  | ~6 fps  | Notes: deblur without downsampling first
* **SAYCAM_large:**        1M 640x480 blur frames    | 400 folders - 2500 each | ~30 fps |
* **SAYCAM_large_deblur:** 200K 640x480 deblur crops | 400 folders - 500 each  | ~6 fps  | Notes: deblur ofc without downsampling first

### Pre-trained Models

* **1M_blur:**                           Dataset: SAYCAM                           | Image Transformations: ALL                        |
* **200K_blur:**                         Dataset: SAYCAM_blur                      | Image Transformations: ALL                        |
* **200K_deblur:**                       Dataset: SAYCAM_deblur                    | Image Transformations: ALL                        |
* **200K_deblur_new:**                   Dataset: SAYCAM_deblur_new                | Image Transformations: ALL                        |
* **200K_blur_full:**                    Dataset: SAYCAM_blur_full                 | Image Transformations: ALL                        |
* **200K_blur_full_:**                   Dataset: SAYCAM_blur_full                 | Image Transformations: ALL except normalization   |
* **200K_deblur_new_full:**              Dataset: SAYCAM_large_deblur              | Image Transformations: ALL                        |
* **200K_deblur_new_full_:**             Dataset: SAYCAM_large_deblur              | Image Transformations: ALL except normalization   |
* **200K_blur_deblur_wo_transforms:**    Dataset: SAYCAM_blur & SAYCAM_deblur_new  | Image Transformations: NONE                       | Notes: blur/deblur_new pairs
* **200K_blur_deblur_w_transforms:**     Dataset: SAYCAM_blur & SAYCAM_deblur_new  | Image Transformations: ALL                        | Notes: blur/deblur_new pairs
