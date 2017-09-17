#!/usr/bin/env sh

TOOLS=./build/tools

# cuda-gdb --args \
$TOOLS/caffe train \
    --solver=//media/hpc2_storage/eustinova/np_pedestrians/experiments/market_np_loss_1e-4_grid_0.01_simple_bilinear_TEST/protos/solver.prototxt \
    --gpu 0
