## Train:

Curiculum for training the WritableArea (parcment) model:
```bash
python3 ./bin/ddp_train_resolution.py -gt_paths ./data/1000_CVCharters/*/*/*/*.seals.crops/*.resolution.gt.json -augment_scale_range '[1.0, 1.0]' ; 
python3 ./bin/ddp_train_resolution.py -gt_paths ./data/1000_CVCharters/*/*/*/*.seals.crops/*.resolution.gt.json  -epochs 200; 
python3 ./bin/ddp_train_resolution.py -gt_paths ./data/1000_CVCharters/*/*/*/*.seals.crops/*.resolution.gt.json -lr 0.0003 -epochs 1000
```

```bash
PYTHONPATH="./" ./bin/ddp_train_resresnet.py -gt_paths ./data/1000_CVCharters/*/*/*/*.crops/*.gt.json -device cpu
```