#!/usr/bin/env bash
wget https://s3.amazonaws.com/wavegan-v1/models/timit.ckpt.index -O model.ckpt.index
wget https://s3.amazonaws.com/wavegan-v1/models/timit.ckpt.data-00000-of-00001 -O model.ckpt.data-00000-of-00001
wget https://s3.amazonaws.com/wavegan-v1/models/timit_infer.meta -O infer.meta
