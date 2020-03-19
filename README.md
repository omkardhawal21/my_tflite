# my_tflite

This repository contains the model whos pretrained weights were used fro transfer Learning.
I refered this medium article for the entire process - https://towardsdatascience.com/deeppicar-part-6-963334b2abe0
I've used non-quantized model.
Tensorflow version 1.14

Please refer pipeline.config file for further reference.

model_ckpt folder contains model checkpoints after 2000 steps.

frozen_inference_graph.pb was exported from above checkpoints.
I've added main_test.py file which was used to test the frozen_inference_graph.pb
Please Relocate main_test.py file at YOUR_PATH/models/research/object_detection (Official github repository- https://github.com/tensorflow/models/tree/master/research/object_detection)
Also give path to slim package when executing above file.

tflite_graph.pb was also exported from model checkpoints.

tflite_graph.pb was coverted to tflite file named sign.tflite.

my issue -
frozen_infeence_graph.pb gives very good result on images.
But when I use tflite file in Android App, it gives no result(no boundin boxex).
In logs it's clearly written that my model processed 0 results in images.
Initially I thought,there is something wrong with APP code. So I used same model but very poorly trained(50 steps) and got few wrong results on screen.
This implies that APP code was right but the model I trained on 2000 steps is giving wrong results.
