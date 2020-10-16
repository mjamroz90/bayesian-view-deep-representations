Here we describe steps to reproduce our experiments with Beta-VAEs and MMD-VAEs. All of the commands described below should be executed from the project's main directory.

## Anaconda environments

There are two conda environments required to conduct all of the experiments described below: ```tf2``` and ```tf_mkl```. The second one (```tf_mkl```) is for running DP-GMM estimation on latent codes. The first one (```tf2```) should be used for all others steps. To create these environments from the ```*.yml``` files (provided with the code), run:
```
conda env create -f [tf2.yml | tf_mkl.yml]
```
Before running any experiment, activate a proper environment by issuing:
```
conda activate [tf2 | tf_mkl]
```

### Training VAE

In order to train a variant of Beta-VAE on a given dataset (either CelebA or Anime), execute command:

```
python src/models/vaes/scripts/train_celeb_vae.py <out_weights_dir> <beta> <dataset> --arch bigger --reg_type <variant>
```
Encoder and decoder parameters from subsequent training iterations will be stored in ```<out_weights_dir>``` directory. The Beta value should be given in the second parameter, ```<dataset>``` can be either 'celeb' or 'anime'. Parameter ```--reg_type``` should be set to 'kl' for standard Beta-VAE, and 'mmd-imq' for MMD-VAE.

### Generating latent codes

With VAE trained, one can generate latent codes for the test part of the CelebA/Anime dataset. This is done by the following command:

```
python src/models/vaes/scripts/generate_latent_codes_on_test_set.py <model_path> <out_codes_arr_file>
```

```<model_path>``` should point to the encoder/decoder parameter files located in ```<out_weights_dir>``` folder (see previous step). Results will be stored in ```<out_codes_arr_file>```. Dataset is deduced from the config file stored (by the training script) in the directory with model parameters.

### Running DP-GMM on latent codes

After generating latent codes (for specific beta) one should run a DP-GMM estimation in exactly the same way as in the case of CNNs.
Keep in mind that such estimation should be performed separately for each beta value.

### Estimating relative entropy values from DP-GMM traces (Beta-VAEs)

Let's assume that folders with DP-GMM traces for evaluated beta values are stored in ```<clustering_results_root_dir>```. In order to estimate relative entropy values, one should run:
```
python src/models/vaes/scripts/clustering/estimate_entropy_from_clustering.py <clustering_results_root_dir> <init_iteration> <step> <out_json_file>
```
Relative entropies for all beta values will be collected in a single JSON file: ```<out_json_file>```. Parameter ```<init_iteration>``` is an integer indicating start of the sequence of Gibbs steps used for entropy estimation (preceding steps are considered brun-in period) and ```<step>``` is an integer indicating how many steps to omit between any two steps used for estimation (chain thinning).

### Calculating component counts from DP-GMM traces (Beta-VAEs)

Component counts for evaluated beta values can be calculated with:
```
python src/models/vaes/scripts/clustering/calculate_cluster_counts.py <clustering_results_root_dir> <init_iteration> <step> <out_json_file>
```
This script works exactly like ```estimate_entropy_from_clustering.py```, except that it extracts component counts from the DP-GMM traces.

### Calculating latent dimensions coupling from DP-GMM traces (Beta-VAEs)

To estimate the degree of latent dimensions coupling (total correlation between dimensions of posterior predictive), run:
```
python src/models/vaes/scripts/clustering/calculate_diagonality_of_representation.py <clustering_results_root_dir> <init_iteration> <step> jtpom <out_json_file>
```
This script works exactly like ```estimate_entropy_from_clustering.py```, except that it calculates KL divergence between the posterior predictive distribution and its product of marginals approximation.

### Generating samples from predictive density

To generate samples using latent codes drawn from the posterior predictive density, run:
```
python src/models/vaes/scripts/clustering/generate_samples_from_clusters.py <vae_model_path> <trace_path> <out_vis_path> --mode [joint_mixture | factorial_mixture] --grid_size 8x8
```
Encoder/decoder parameters are loaded from ```<vae_model_path>``` file. DP-GMM components structure is read from ```<trace_path>``` CGS step. Generated samples are saved in ```<out_vis_path>``` folder. Last argument indicate whether latent codes should be drawn from the posterior predictive (```joint_mixture```) or its factorial mixture approximation (```factorial_mixture```).

