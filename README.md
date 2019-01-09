This repository contains a PyTorch-based setup to train a simulated self-driving car, on parts based on the project with Elias Eulig (elias.eulig@stud.uni-heidelberg.de).
---
# README
---
## This Repository contains all code used in our Self-Driving Car IAI 2017 project.
### It is self-contained and uses only relative pathing to allow for easy replication.
---

To successfully use all that this repository contains, you need to:

1. Download the Udacity Self-driving car simulator at [Link to Simulator Repo](https://github.com/udacity/self-driving-car-sim)

2. This repository is build on Python 2.7, PyTorch 0.3 and other packages that come directly with a quick miniconda installation. 
   To run the automatic driving however, several other packages are required. If missing, simply `conda install <required_package>`.

---

The structure of this repository looks something like:

#### Folders:

1. `Data(.zip)`:		The data that was used for training and testing. Contains images from driving on the sea and jungle track twice in each direction, respectively.

2. `Visualisations_and_Plots`:	histograms of road properties s.a. steering angle distributions, feature images computed by the network during driving and other visualisations.

3. `Results(.zip)`:	Contains all trained network weights and logging files to run a pretrained driver.


#### Scripts:
There are several files that need explaining, namely:

1. `prediction_network.py`: 	This script contains the modular network classed we used to build our PyTorch Network.   

2. `helper_functions.py`:		Utility functions that are used in most other scripts. Contains the Training PyTorch Dataset, Logging utilities and other stuff.  

3. `training.py`: 				Training script to fully train the specified network structure.  

4. `run_automatic_driving.py`:	Script performing communication with localhost set up by the Udacity Simulator. Runs the network-based driving.   

5. `summarize.py, test_performance_test.py`: Use these scripts to summarize the training runs by learning curves and other summaries and check the performance on test-sets of choice.  

6. `make_videos.py`:			Convert saved feature map data to nice movie frames.   

---

Examples on how to run each respective script from command line and useful arguments that can be passed can be shown by typing `python <script_name> --help`. The most important ones are however shown here:

_Running network training (Note that you can omit most arguments if you want to run the standard network)_:
```
python training.py --cuda --augment flip jitter angle_jitter shadow --tvsplit 0.7 --all --bs 8 --n_epochs 100 --lr 0.0003 --use_BN --force_shape None 110 --no_center_crop --gamma 0.2 --tau 30 --cat_name full_conv_best_4
				   --make_full_conv --n_filters 24 36 48 64 64 92 --kernel_sizes 5 5 5 3 3 3 --strides 2 2 2 2 1 1 --conv_dropout 0.3 0.3 0.3 0.3 0.3 0 --get_test_perf
```

_Running automatic driving_:
```
python run_automatic_driving.py --cuda --base_folder <name_of_folder_where_you_save_training_information_to> --speed <car_driving_speed_0_to_30> --extract_features
```

_Summarize Training Runs_:
```
python summarize.py --log --inc_test_loss --inc_emd --set_baselines
                    --base_folder <path_to_folder_with_multiple_runs> --save_plot here
```

_Test Test performance_:
```
python test_performance_tester.py --cuda --show_hists --save
```
