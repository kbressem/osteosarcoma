# Description of files in `scripts`

## Files
1. [train.py](train.py)
2. [tune.py](tune.py)
3. [scheduler.py](scheduler.py)
4. [infer.py](infer.py)

## train.py
> start a training

Start training a model. For example, to train a model for binary segmentation (kidney + tumors + cysts against background) on CT images run:  

```python
python train.py --config ../configs/CT/binary.yaml
```

## tune.py
> Run hyperparameter tuning

```python
python train.py --config ../configs/CT/binary.yaml
```
 
## scheduler.py
> Python wrapper to push multiple jobs to slurm

Schedule a training
```python
python scheduler.py \
    --train \
    --config ../configs/binary.yaml \
```

Schedule hyperparameter tuning on 4 single nodes
```python
python scheduler.py \
    --tune \
    --config ../configs/binary.yaml \
    --jobs 4 \
```


## infer.py
> run inference on multiple files

Run model inference on single or multiple files. 

To run prediction on CT images (binary segmentation only) run: 

```python
python infer.py \
    --config ../configs/binary.yaml \
    --model ../models/binary.pt \
    --files /workspace/path/to/image/ \
    --input_postfix image.nii --output_postfix pred_binary 
```
