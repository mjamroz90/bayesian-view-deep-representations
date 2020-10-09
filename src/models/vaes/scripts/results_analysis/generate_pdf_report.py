import argparse
import os
import os.path as op

from fpdf import FPDF

from utils.reports import pdf_generation
from utils import fs_utils


def list_clustering_result_dirs(root_results_dir):
    return [op.join(root_results_dir, p) for p in os.listdir(root_results_dir)
            if op.isdir(op.join(root_results_dir, p)) and p.startswith("celeb_vae")]


def put_clustering_results_to_doc(clustering_result_dirs, metadata_dict):
    doc_obj = FPDF()
    for i in range(0, len(clustering_result_dirs), 4):
        doc_obj.add_page()

        clustering_result_batch = clustering_result_dirs[i: i + 4]
        pdf_generation.prepare_header(doc_obj, "BETA-VAE", 12)
        clustering_metadata_batch = [metadata_dict[op.basename(p)] for p in clustering_result_batch]

        img_grid_paths = [op.join(cd, "clusters_dynamics.png") for cd in clustering_result_batch]
        img_grid_headers = ["LATENT DIM - %d, BETA - %s" % (data_dim, str(b))
                            for data_dim, b in clustering_metadata_batch]

        pdf_generation.insert_images_grid(doc_obj, img_grid_paths, img_grid_headers)

    return doc_obj


def main():
    args = parse_args()
    clustering_result_dirs = list_clustering_result_dirs(args.root_results_dir)
    metadata_dict = fs_utils.read_json(op.join(args.root_results_dir, 'metadata.json'))

    doc = put_clustering_results_to_doc(clustering_result_dirs, metadata_dict)
    doc.output(args.out_pdf_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_results_dir')
    parser.add_argument('out_pdf_file')
    return parser.parse_args()


if __name__ == '__main__':
    main()
