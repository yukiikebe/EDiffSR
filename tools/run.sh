xhost +local:root

docker run -it --entrypoint /bin/bash -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/yuki/research/EDiffSR:/workspace \
    -v /dev:/dev --privileged -p 6006:6006\
    -e DISPLAY=$DISPLAY --gpus all yuki/super_resolution:latest


# python train.py -opt=options/train/refusion.yml
