Here we describe steps to replicate our experiments with CNNs. All of the commands described below should be executed from the project's main directory. dataset,

## Anaconda environments

There are two Anaconda environments required to conduct CNN experiments: ```reproduce``` and ```tf_mkl```. To create these environments from the ```*.yml``` files (provided with the code), run:
```
conda env create -f [reproduce.yml | tf_mkl.yml]
```
Before running any command described below, activate a proper environment by executing:
```
conda activate [reproduce | tf_mkl]
```
The ```reproduce``` environment is used only to train CNN models and extract activations from them. All other steps should be done in the ```tf_mkl``` environment.

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

For each of these 3 variant, a set of 50 instances (with different initial weight values) will be trained. By default training will be carried out for 30 epochs (one can change that by setting ```[epochs_num]``` argument). Argument ```<net_name>``` can be set to one of the values defined under NN_ARCHITECTURES in ```src/settings.py```. All training artifacts are stored in ```mlruns``` directory created in the current directory. Each variant is stored in a following folder structure:
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
python scripts/extract_activations_on_dataset.py <net_name> <models_dir> <dataset> <out_activations_dir> --agg_mode both
```
After executing the above command, folder ```out_activations_dir``` will contain activations collected across all 50 net instances, as well as activations for each instance individually. The structure should look like this:
```
<out_activations_dir>
   - avg_pooling_0_acts/
   - avg_pooling_0_acts.npy
   ...
   - avg_pooling_10_acts/
   - avg_pooling_10_acts.npy
```

### Dimensionality reduction

With activation vectors calculated, one can start reducing their dimensionality using SVD. Our scripts are prepared for parallel execution under SLURM cluster management system. However, one can treat SLURM scripts just like ```bash``` scripts and execute them one by one from a console. There are special comments inside SLURM scripts that are interpreted by SLURM to allocate required hardware resources - this comments indicate how much memory is (roughly) needed to execute the command and how many CPU cores it can efficiently consume. SLURM tasks are submitted with ```sbatch``` command, which takes as an argument a SLURM script that contains commands to execute. Again: SLURM is **not** required to run these tasks - one can instead run them one by one from a console, just like ```bash``` scripts.

A following command prepare ```sbatch``` scripts for reducing the dimensionality of activation vectors: 
```
python scripts/datamunging/sbatch_cmds/prepare_eigact_svd_reduction_cmds.py <act_dir_1> <act_dir_2> ... <act_dir_n> <out_root_dir> <out_sbatch_cmds_dir> <dim_config_json> <suffix>
```
One can list multiple directories with extracted activations (```<act_dir_*>```). ```<out_root_dir>``` is a directory under which ```sbatch``` scripts will store results. The scripts themselves will be stored in ```<out_sbatch_cmds_dir>``` directory, which has a following structure:
```
<out_sbatch_cmds_dir>
   - 0.sh
   - 1.sh
   - 2.sh
   ...
   - <10>.sh
```
```<dim_config_json>``` is a JSON file containing mapping between layer indices and the number of dimensions. It should be ```ld_11.json``` for networks with 11 convolutions and ```ld_8.json``` for networks with 8 convolutions.

Dimensionality reduction scripts generate output in a following way:
- Let's assume that there are three folders with input activations: ```true_labels```, ```true_labels_aug``` ```random_labels```
- Results will then be written to following folders:
```
<out_root_dir>
   - true_labels_<suffix>/
   - true_labels_aug_<suffix>/
   - random_labels_<suffix>/
```
each one with a content like:
```
<out_root_dir>/true_labels_ld
   - avg_pooling_0_acts_eigact.npy
   - avg_pooling_1_acts_eigact.npy
   ...
   - avg_pooling_10_acts_eigact.npy
```
As a suffix one can use e.g.: ```ld``` (which is assumed by the result analysis scripts).

## DP-GMM model estimation

Similarly to the previous step, DP-GMM model estimation is done with SLURM scripts (which can be run sequentially from a console, without SLURM). In order to prepare necessary scripts, one has to run:
```
python scripts/datamunging/sbatch_cmds/prepare_eigact_clustering_cmds.py <eigact_dir_1> <eigact_dir_2> ... <eigact_dir_n> <out_results_dir> <out_sbatch_cmds_dir>
```
Each input folder ```<eigact_dir_*>``` must contain activations for all layers (it could be e.g. ```true_labels_ld``` directory from the previous step).

Assuming that input directories are ```<path_to_eigenacts>/true_labels_ld```, ```<path_to_eigenacts>/true_labels_aug_ld```, ```<path_to_eigenacts>/random_labels_ld```, the structure of the output folders (after running the SLURM script) will look like:
```
<out_results_dir>
   - true_labels_ld/
   - true_labels_aug_ld/
   - random_labels_ld/
```
and each output folder will contain trace of Gibbs Sampling steps for each network layer (i.e clusters assignments, parameters of various probability distributions, etc.).

## Summarizing DP-GMM traces

In order to extract relevant information from clustering traces, one can execute a following command:
```
python scripts/results_analysis/fetch_eigact_clustering_results.py <clustering_results_dir> <out_results_dir>
```
```<out_results_dir>``` will then contain plots summarizing component counts and log likelihoods during Gibbs sampling. Plots will be stored in a folder structure like:
```
<clustering_results_dir>
   - avg_pooling_0_acts_eigact_results
   - avg_pooling_1_acts_eigact_results
   - ...
   - avg_pooling_10_acts_eigact_results
```

## Calculating entropy values

In order to estimate relative entropy values from DP-GMM traces, execute a following command:
```
python scripts/results_analysis/estimate_entropy_from_clustering.py <clustering_results_root_dir> <init_iteration> <step> --entropy_type relative
```
Where ```<init_iteration>``` is an integer indicating beginning of the sequence of Gibbs steps used for entropy estimation (preceding steps are considered brun-in period) and ```<step>``` is an integer indicating how many steps to omit between any two steps used for estimation (chain thinning). The script will calculate mean, minimum and maximum value over the sampled Gibbs steps.

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
