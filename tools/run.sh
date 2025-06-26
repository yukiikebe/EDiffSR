xhost +local:root
export WANDB_API_KEY=60c883aff30a57af75e35c552172dbd07a2c9a2c

docker run --shm-size=16g -it --entrypoint /bin/bash -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/yuki/research/EDiffSR:/workspace \
    -e WANDB_API_KEY=$WANDB_API_KEY\
    -e DISPLAY=$DISPLAY --gpus all yuki/super_resolution:latest


# python train.py -opt=options/train/refusion.yml

# --gpus '"device=2,3"'

# for gdal
# docker run -it --rm -v $(pwd):/workspace ghcr.io/osgeo/gdal:ubuntu-full-latest