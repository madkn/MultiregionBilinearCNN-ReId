#!/usr/bin/env sh

python ./tools/extra/parse_log.py \
    //media/hpc2_storage/eustinova/np_pedestrians/experiments/market_np_loss_1e-4_grid_0.01_simple_bilinear_TEST/logs/log.txt \
    //media/hpc2_storage/eustinova/np_pedestrians/experiments/market_np_loss_1e-4_grid_0.01_simple_bilinear_TEST/logs

python ./tools/extra/plot_log.py \
    //media/hpc2_storage/eustinova/np_pedestrians/experiments/market_np_loss_1e-4_grid_0.01_simple_bilinear_TEST/logs/log.txt.$1 -f $2 -r $3
