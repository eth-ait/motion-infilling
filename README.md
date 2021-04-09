# Convolutional Autoencoders for Human Motion Infilling
This repository contains the code to [our paper](https://ait.ethz.ch/projects/2020/motion_infilling/) published at 3DV 2020. 

# Data
This project uses data provided by Daniel Holden et al.'s paper A Deep Learning Framework for Character Motion Synthesis and Editing, which can be downloaded [here](http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing). We preprocess the raw data and create training/validation splits with the scripts provided in [data-preprocessing](data-preprocessing). For this to work, you will need some helper functions provided in Daniel Holden's code base in the `motion` module. The code base can be downloaded [here](http://theorangeduck.com/media/uploads/other_stuff/motionsynth_code.zip).

# Code
## Environment
This project is a bit older and hence the libraries it uses are too. It was tested with Python 3.5, CUDA 8 and TensorFlow 0.12. Tensorflow was installed via

```
(On Windows)
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-win_amd64.whl

(On Linux)
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
```

A list of all dependencies is provided in `requirements.txt`. Lastly, make sure the folder `tbase` is on the python path. If you are using PyCharm, the quickest is to mark the folders `infilling` and `tbase` as "sources root".

## Quantitative Evaluation
All necessary code for training and evaluation is provided in [infilling](infilling). You need to adjust paths in two places:
 - `get_data_path()` in `flags_parser.py`: return the path where the preprocessed data is located.
 - `get_checkpoints_path()` in `flags_parser.py`: return the path where pre-trained models are stored, currently set to `./pretrained-models`.

Under [pretrained-models](pretrained-models) we provide the models mentioned in the paper. The mapping is as follows:

| Name in Paper | Path to Model |
| --- | --- |
| Holden et al. | [pretrained-models/HoldenCAE/run_003](pretrained-models/HoldenCAE/run_003) |
| Vanilla AE | [pretrained-models/VanillaCAE/run_028](pretrained-models/VanillaCAE/run_028) |
| Ours (60) | [pretrained-models/VGG/run_020](pretrained-models/VGG/run_020) |
| Ours (curr.) | [pretrained-models/VGG/run_031](pretrained-models/VGG/run_031) |

To evaluate a pre-trained model, use the `evaluate.py` script. For example, to re-create the last row of Table 1 in the main paper use:

```
python evaluate.py --split validation --runs VGG/31f
python evaluate.py --split validation --runs VGG/31fp --perturbator column --perturbation_size [60]
python evaluate.py --split validation --runs VGG/31fp --perturbator column --perturbation_size [120]
```

The syntax of the `--runs` parameter is as follows. `VGG/31` means that we want to evaluate the model under `pretrained-models/VGG/run_031`. `f` signals that foot contacts should be removed from the input data (we do not use them in our method) and `p` applies the perturbations. If the `p` is missing, no perturbations are applied.

Hence, to reproduce row 2 in Table 1, you can use:
```
python evaluate.py --split validation --runs HoldenCAE/3
python evaluate.py --split validation --runs HoldenCAE/3p --perturbator column --perturbation_size [60]
python evaluate.py --split validation --runs HoldenCAE/3p --perturbator column --perturbation_size [120]
```

To run the linear interpolation baseline, use `--lerp`.

## Qualitative Evaluation
The scripts starting with `showcase_*.py` were used to create the results shown in the video. Please refer to those scripts directly for more details. Very simple visualizations are available via `visualize.py`.

## Training
The `main.py` script lets you re-train models from scratch. Please refer to the file `config.txt` which is available for each pretrained model to see which configuration parameters were used. Models are implemented in `models.py` and some layer implementations can be found in `ops.py`.

# Citation
If you use this code, please cite
```
@inproceedings{MotionInfilling,
  author={Manuel {Kaufmann} and Emre {Aksan} and Jie {Song} and Fabrizio {Pece} and Remo {Ziegler} and Otmar {Hilliges}},
  booktitle={2020 International Conference on 3D Vision (3DV)}, 
  title={Convolutional Autoencoders for Human Motion Infilling}, 
  year={2020},
  pages={918-927},
  doi={10.1109/3DV50981.2020.00102}}
```

# Contact Information
For questions or problems please create an issue or contact [manuel.kaufmann@inf.ethz.ch](mailto:manuel.kaufmann@inf.ethz.ch).
