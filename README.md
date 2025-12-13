Commands:

pip install -r requirements.txt (include torch, torchvision, pyyaml, numpy, wandb etc.,)

Make sure wandb login is done before running the script to log data.

Baseline training:
    # saves: ./artifacts/baseline_fp32_best.pth
    python train.py --config config.yaml 

Compression run:
    # saves final compressed .qmod and .state_dict.pth in ./artifacts
    1. Modify configuration in the config.yaml file
    2. python compress.py --config config.yaml --fp32 ./artifacts/baseline_fp32_best.pth
    3. To enable wandb logging set use_wandb: true in config.yaml or pass --use_wandb to compress.py. The code will attempt to log artifacts but will not fail if wandb is not configured.
    4. Other model paramters can also be changes in config.yaml file itself.

Test run:
    # or use --qmod to load compressed artifact if you added that option
    1. Copy the compressed model from ./artifacts/mobilenetv2_compressed_...state_dict.pth if compress.py was run and paste in current    directory as mobilenetv2_compressed_state_dict.pth
    2. python test.py --state mobilenetv2_compressed_state_dict.pth

NOTE: Sample run artifacts are placed inside the artifacts folder. The actual model files are in the current directory itself.
    1. best_mobilenetv2_epoch99_val83.84.pth -> BASELINE MODEL
    2. mobilenetv2_compressed_model.qmod -> CUSTOM COMPRESSED REPRESENTATION OF QUANTIZED WEIGHTS
    3. mobilenetv2_compressed_state_dict.pth -> MODEL PARAMETERS IN FP32 MODEL

    