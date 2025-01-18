## Installation

1. Install `torch>=2.0.0`.
2. Install other pip packages via `pip3 install -r requirements.txt`.
3. Prepare the [ImageNet](http://image-net.org/) dataset
    <details>
    <summary> assume the ImageNet is in `/path/to/imagenet`. It should be like this:</summary>

    ```
    /path/to/imagenet/:
        train/:
            n01440764: 
                many_images.JPEG ...
            n01443537:
                many_images.JPEG ...
        val/:
            n01440764:
                ILSVRC2012_val_00000293.JPEG ...
            n01443537:
                ILSVRC2012_val_00000236.JPEG ...
    ```
   **NOTE: The arg `--data_path=/path/to/imagenet` should be passed to the training script.**
    </details>

5. (Optional) install and compile `flash-attn` and `xformers` for faster attention computation. Our code will automatically use them if installed. See [models/basic_var.py#L15-L30](models/basic_var.py#L15-L30).


## Training Scripts

To train VAR-{d16, d20, d24, d30, d36-s} on ImageNet 256x256 or 512x512, you can run the following command:
```shell
# d16, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1
# d20, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=20 --bs=768 --ep=250 --fp16=1 --alng=1e-3 --wpe=0.1
# d24, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=24 --bs=768 --ep=350 --tblr=8e-5 --fp16=1 --alng=1e-4 --wpe=0.01
# d30, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=30 --bs=1024 --ep=350 --tblr=8e-5 --fp16=1 --alng=1e-5 --wpe=0.01 --twde=0.08
# d36-s, 512x512 (-s means saln=1, shared AdaLN)
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=36 --saln=1 --pn=512 --bs=768 --ep=350 --tblr=8e-5 --fp16=1 --alng=5e-6 --wpe=0.01 --twde=0.08
```
A folder named `local_output` will be created to save the checkpoints and logs.
You can monitor the training process by checking the logs in `local_output/log.txt` and `local_output/stdout.txt`, or using `tensorboard --logdir=local_output/`.

If your experiment is interrupted, just rerun the command, and the training will **automatically resume** from the last checkpoint in `local_output/ckpt*.pth` (see [utils/misc.py#L344-L357](utils/misc.py#L344-L357)).