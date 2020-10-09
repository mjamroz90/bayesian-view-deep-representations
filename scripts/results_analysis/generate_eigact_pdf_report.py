import argparse
import os
import os.path as op

from utils import fs_utils
from utils.logger import log
from utils.reports import pdf_generation


def list_result_dirs(result_root_dir):
    layers_result_dirs = [op.join(result_root_dir, p) for p in os.listdir(result_root_dir)
                          if op.isdir(op.join(result_root_dir, p)) and p.endswith(('_results',))]
    layers_result_dirs_sorted = sorted(layers_result_dirs, key=lambda d: int(op.basename(d).split('_')[2]))
    return layers_result_dirs_sorted


def list_layers_dirs(result_root_dir, dir_structure_mode, true_mode):
    def __find_true_random(dir_name):
        sub_dirs = [d for d in os.listdir(dir_name) if op.isdir(op.join(dir_name, d))]
        t_dir, r_dir = None, None
        for sd in sub_dirs:
            start_true_prefix = 'true_labels_ld' if true_mode == 'no_aug' else 'true_labels_aug_ld'
            if sd.startswith(start_true_prefix) and sd.endswith('results'):
                t_dir = op.join(dir_name, sd)

            if sd.startswith('random') and sd.endswith('results'):
                r_dir = op.join(dir_name, sd)

        return t_dir, r_dir

    if dir_structure_mode == 'train_test':
        train_root_dir, test_root_dir = op.join(result_root_dir, 'train'), op.join(result_root_dir, 'test')
        train_true_dir, train_random_dir = __find_true_random(train_root_dir)
        test_true_dir, test_random_dir = __find_true_random(test_root_dir)

        true_result = {'train': list_result_dirs(train_true_dir), 'test': list_result_dirs(test_true_dir)}
        random_result = {'train': list_result_dirs(train_random_dir), 'test': list_result_dirs(test_random_dir)}

        result = (true_result, random_result)
    else:
        true_dir, random_dir = __find_true_random(result_root_dir)
        result = (list_result_dirs(true_dir), list_result_dirs(random_dir))

    return result


def put_layers_results_to_document(true_results_layers_info, random_results_layers_info, entropy_mode):
    from fpdf import FPDF

    def __fetch_layer_info_for_results_info(results_info, index):
        if 'train' in results_info and 'test' in results_info:
            train_path = results_info['train']['paths'][index]
            test_path = results_info['test']['paths'][index]
            return {'train': (train_path, results_info['train']['metadata'][op.basename(train_path)]),
                    'test': (test_path, results_info['test']['metadata'][op.basename(test_path)])}
        else:
            path = results_info['paths'][index]
            return {'test': (path, results_info['metadata'][op.basename(path)]), 'train': None}

    doc = FPDF()
    if 'train' in true_results_layers_info:
        layers_num = len(true_results_layers_info['train']['paths'])
    else:
        layers_num = len(true_results_layers_info['paths'])

    for layer_index in range(layers_num):
        true_index_info = __fetch_layer_info_for_results_info(true_results_layers_info, layer_index)
        random_index_info = __fetch_layer_info_for_results_info(random_results_layers_info, layer_index)

        doc.add_page()
        img_grid_paths, img_grid_headers = prepare_img_grid_paths_and_titles(true_index_info, random_index_info,
                                                                             entropy_mode)

        pdf_generation.prepare_header(doc, "Layer %d" % layer_index, 12)
        pdf_generation.insert_images_grid(doc, img_grid_paths, img_grid_headers)

    return doc


def prepare_img_grid_paths_and_titles(true_index_info, random_index_info, entropy_mode):
    true_test_img_path = op.join(true_index_info['test'][0], 'clusters_dynamics.png')
    random_test_img_path = op.join(random_index_info['test'][0], 'clusters_dynamics.png')

    true_test_metadata = true_index_info['test'][1]
    random_test_metadata = random_index_info['test'][1]

    true_test_entropy = read_entropy_val(true_index_info['test'][0], entropy_mode)
    random_test_entropy = read_entropy_val(random_index_info['test'][0], entropy_mode)

    img_grid_paths = [true_test_img_path, random_test_img_path, None, None]
    img_grid_headers = ["TRUE-TEST (%d, %.3f), ENTROPY: %.3f" % (true_test_metadata[1], true_test_metadata[0],
                                                                 true_test_entropy),
                        "RANDOM-TEST (%d, %.3f), ENTROPY: %.3f" % (random_test_metadata[1], random_test_metadata[0],
                                                                   random_test_entropy), None, None]

    if 'train' in true_index_info:
        true_train_img_path = op.join(true_index_info['train'][0], 'clusters_dynamics.png')
        true_train_metadata = true_index_info['train'][1]
        true_train_entropy = read_entropy_val(true_index_info['train'][0], entropy_mode)
        img_grid_paths[2] = true_train_img_path
        img_grid_headers[2] = "TRUE-TRAIN (%d, %.3f), ENTROPY: %.3f" % (true_train_metadata[1], true_train_metadata[0],
                                                                        true_train_entropy)

    if 'train' in random_index_info:
        random_train_img_path = op.join(random_index_info['train'][0], 'clusters_dynamics.png')
        random_train_metadata = random_index_info['train'][1]
        random_train_entropy = read_entropy_val(random_index_info['train'][0], entropy_mode)
        img_grid_paths[3] = random_train_img_path
        img_grid_headers[3] = "RANDOM-TRAIN (%d, %.3f), ENTROPY: %.3f" % (random_test_metadata[1],
                                                                          random_train_metadata[0], random_train_entropy)

    return img_grid_paths, img_grid_headers


def read_entropy_val(layer_results_path, entropy_mode):
    if entropy_mode == 'differential':
        entropy_path = op.join(layer_results_path, 'entropy.json')
        if not op.exists(entropy_path):
            entropy_path = op.join(layer_results_path, 'entropy_diff.json')
    else:
        entropy_path = op.join(layer_results_path, 'entropy_relative.json')

    if op.exists(entropy_path):
        entropy_dict = fs_utils.read_json(entropy_path)
        return float(entropy_dict['mean'])
    else:
        return None


def infer_dir_structure_mode(root_dir):
    if op.exists(op.join(root_dir, 'train')) and op.exists(op.join(root_dir, 'test')):
        return 'train_test'
    else:
        return 'test_only'


@log
def add_metadata_info(results_layers_dirs, mode):
    def __add_suffix(meta_dict):
        return {"%s_results" % k: v for k, v in meta_dict.items()}

    if mode == 'train_test':
        train_metadata_file = op.join(op.dirname(results_layers_dirs['train'][0]), 'metadata.json')
        test_metadata_file = op.join(op.dirname(results_layers_dirs['test'][0]), 'metadata.json')

        add_metadata_info.logger.info("Mode: train_test, reading metadata for train/test from paths: %s / %s" %
                                      (train_metadata_file, test_metadata_file))
        train_metadata = fs_utils.read_json(train_metadata_file)
        test_metadata = fs_utils.read_json(test_metadata_file)
        return {'train': {'paths': results_layers_dirs['train'], 'metadata': __add_suffix(train_metadata)},
                'test': {'paths': results_layers_dirs['test'], 'metadata': __add_suffix(test_metadata)}}
    else:
        metadata_file = op.join(op.dirname(results_layers_dirs[0]), 'metadata.json')

        add_metadata_info.logger.info("Mode: test_only, reading metadata from path: %s" % metadata_file)
        metadata = fs_utils.read_json(metadata_file)

        return {'paths': results_layers_dirs, 'metadata': __add_suffix(metadata)}


def main():
    args = parse_args()
    mode = infer_dir_structure_mode(args.results_root_dir)

    true_results_layers_dirs, random_results_layers_dirs = list_layers_dirs(args.results_root_dir, mode, args.true_mode)

    true_results_layers_info = add_metadata_info(true_results_layers_dirs, mode)
    random_results_layers_info = add_metadata_info(random_results_layers_dirs, mode)

    doc_to_save = put_layers_results_to_document(true_results_layers_info, random_results_layers_info,
                                                 args.entropy_mode)
    doc_to_save.output(args.out_pdf_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_root_dir')
    parser.add_argument('out_pdf_file')
    parser.add_argument('--true_mode', choices=['aug', 'no_aug'], default='no_aug')
    parser.add_argument('--entropy_mode', choices=['relative', 'differential'], default='differential')
    return parser.parse_args()


if __name__ == '__main__':
    main()
