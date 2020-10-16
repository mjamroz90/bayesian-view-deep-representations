Here we describe steps to reproduce our experiments with CNNs. All of the commands described below should be executed from the project's main directory.

## Anaconda environments

There are three Anaconda environments required to conduct CNN experiments: ```reproduce```, ``tf_mkl``` and ```tf2```.
To create these environments from the ```*.yml``` files (provided with the code), run:
```
conda env create -f [reproduce.yml | tf_mkl.yml | tf2.yml]
```
Before running any command described below, activate a proper environment by executing:
```
conda activate [reproduce | tf_mkl | tf2]
```
The ```reproduce``` environment is used only for training CNN models and extracting activations from them.
The ```tf_mkl``` environment is for estimating DP-GMM posteriors (CGS algorithm), and ```tf2``` is used to estimate
entropy values.

## Training networks
To train a set of networks with a given architecture (on a given dataset), run:
```
python src/experiments/scripts/run_v1.py <net_name> <seed> <dataset> [epochs_num]
```
This script trains the ```net_name``` network in 3 variants:
- true labels with augmentation,
- true labels without augmentation,
- permuted (random) labels.
Parameter ```dataset``` can take two values: ```cifar``` or ```imagenet```.

For each of these 3 variants, a set of 50 instances (with different initial weight values) will be trained. By default, training will be carried out for 30 epochs (one can change that by setting ```[epochs_num]``` argument). Argument ```<net_name>``` can be set to one of the values defined under NN_ARCHITECTURES in ```src/settings.py```. All training artifacts are stored in ```mlruns``` directory created in the current directory. Each variant is stored in a following folder structure:
```
mlruns
  - 0/
  - 1/
  - 2/
  ...
  - <id>
```
Contents of every folder (i.e. trained net variant) can be read from ```<id>/meta.yaml``` file.

## Data preparation

### Model extraction

Trained models must to be extracted to a separate folder (from where they will be read by subsequent scripts). A following command extracts models from folder ```1``` inside ```mlruns```:
```
python scripts/datamunging/extract_models_from_experiment <path_to_mlruns>/1 <out_models_dir>
```
It will copy all 50 models into ```<out_models_dir>```  directory:
```
<out_models_dir>
  - model_0.pth
  - model_1.pth
  ...
  - model_49.pth
```

### Calculating activations

A following command will calculate activation vectors for models stored in ```<models_dir>``` folder:
```
python scripts/extract_activations_on_dataset.py <net_name> <models_dir> <dataset> <out_activations_dir> --agg_mode aggregate
```
After executing the above command, folder ```out_activations_dir``` will contain activations collected across all 50 net instances. The structure should look like this:
```
<out_activations_dir>
   - avg_pooling_0_acts.npy
   ...
   - avg_pooling_10_acts.npy
```

### Dimensionality reduction

With activation vectors calculated, one can start reducing their dimensionality using SVD. This is done by: ```scripts/do_dim_reduction_with_svd.py``` script:
```
python scripts/do_dim_reduction_with_svd.py <in_array_file> <out_array_file> --axis 0 --num_features <dim>
```
Argument ```<in_array_file>``` is a path to a file with activation vectors (eg. ```avg_pooling_10_acts.npy```).
Results (i.e. reduced-dimensionality representations) are saved in ```<out_array_file>``` (eg. ```avg_pooling_10_acts_ld.npy```), the output
dimensionality is equal to ```<dim>```. Per-layer dimensionality used in our experiments can be found in: ```ld_8.json``` or
```ld_11.json``` (for newtorks with 8 and 11 convolutional layers, respectively).

This script needs to be invoked for each layer iand each network variant (true labels, true labels with augmentation, random labels, etc).

## DP-GMM model estimation

DP-GMM model estimation can be done with ```scripts/do_clustering_on_npy_arr.py``` script. This script is called in a following way:
```
python scripts/do_clustering_on_npy_arr.py <in_activations_npy_file> <out_dir> shared --max_clusters_num <init_clusters_num> --iterations_num 600 --init_type init_data_stats
```

```<in_activations_npy_file>``` is a path to numpy array with neural activations (with reduced dimensionality - see previous step),
```<init_clusters_num>``` is the initial number of components that Collapsed Gibbs Sampler assigns to data points.
In our experiments we use: ```<init_clusters_num> = int(np.log2(1 + n_samples)```, where
```n_samples``` is the size of the dataset. For instance, a dataset with activations from 50 trained networks and a
layer with 512 convolutional filters will have a size of ```25600 = 50*512```, so ```<init_clusters_num>``` should be set to ```14```.
Traces produced by the Gibbs sampler execution will be stored in ```<out_dir>``` directory. Each trace contains various
quantities, like clusters assignments, parameters of various probability distributions, etc.

## Summarizing DP-GMM traces

In order to extract relevant information from clustering traces, one can execute a following command:
```
python scripts/results_analysis/fetch_eigact_clustering_results.py <clustering_results_dir> <out_results_dir>
```
```<out_results_dir>``` will then contain plots summarizing component counts and log likelihoods during Gibbs sampling. Plots will be stored in a folder with structure like:
```
<clustering_results_dir>
   - avg_pooling_0_acts_eigact_results
   - avg_pooling_1_acts_eigact_results
   - ...
   - avg_pooling_10_acts_eigact_results
```

## Calculating relative entropy values

In order to estimate relative entropy values from DP-GMM traces, execute a following command:
```
python scripts/results_analysis/estimate_entropy_from_clustering.py <clustering_results_root_dir> <init_iteration> <step> --entropy_type relative
```
Where ```<init_iteration>``` is an integer indicating start of the sequence of Gibbs steps that will be used for relative entropy estimation (preceding steps are considered burn-in period) and ```<step>``` is an integer indicating how many steps to omit between any two steps used for estimation (chain thinning). The script will calculate mean, minimum and maximum value over the sampled Gibbs steps.

This script assumes that ```<clustering_results_root_dir>``` contains clustering results for a single network in up to three variants: true labels, true labels + image augmentation and random labels. These results should be stored in the following folders:
```
<clustering_results_root_dir>
   - true_labels_ld
   - true_labels_aug_ld
   - random_labels_ld
```
Missing folders are ignored.

This script will save all calculated entropy values to a JSON file: ```<clustering_results_dir>/entropy_relative.json```.

## Calculating component counts

In order to calculate component counts from DP-GMM traces, execute a following command:
```
python scripts/results_analysis/calculate_cluster_counts.py <clustering_results_root_dir> <init_iteration> <step>
```
This script works similarly to ```estimate_entropy_from_clustering.py``` (```<init_iteration>``` and ```<step>``` have exact same semantics). Results are stored in a JSON file: ```<clustering_results_dir>/clustering_counts.json```.

## Other

### Config files provided with the sources

- ```cifar_corrupted_labels.json``` - fixed set of permuted labels (used when training with random labels) for CIFAR dataset,
- ```imagenet_corrupted_labels.json``` - fixed set of permuted labels for Mini-ImageNet dataset,
- ```ld_11.json```, ```ld_8.json``` - layer/dimensionality mappings.
 
### Conda environments

- ```reproduce.yml``` - conda environment definition for network training and activation extraction,
- ```tf_mkl.yml```  - conda environment definition for estimating DP-GMM.
- ```tf2.yml``` - conda environment for VAE-related experiments
