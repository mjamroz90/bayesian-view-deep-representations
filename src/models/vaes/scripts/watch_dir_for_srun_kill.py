import argparse
import os
import os.path as op
import re
import threading
import subprocess
from datetime import timedelta
import time
import signal

from utils import logger
from utils import fs_utils


class ProgramKilled(Exception):
    pass


def signal_handler(signum, frame):
    raise ProgramKilled


@logger.log
class PeriodicJob(threading.Thread):

    def __init__(self, interval, training_dir, out_sbatch_dir):
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.interval = interval
        self.training_dir = training_dir
        self.out_sbatch_dir = out_sbatch_dir

        self.execute = self.prepare_execute_job
        self.times_executed = 0

    def run(self):
        self.execute()
        self.logger.info("Executed for the first time")
        self.times_executed += 1

        while not self.event.wait(self.interval.total_seconds()):
            self.logger.info("Before executing task %s time" % self.times_executed)
            self.execute()
            self.logger.info("After executing task %s time" % self.times_executed)
            self.times_executed += 1

    def prepare_execute_job(self):
        last_model_path = get_last_model_path_from_dir(self.training_dir)
        self.logger.info("Fetched model from path %s as the last one from training" % last_model_path)
        train_config = fs_utils.read_json(op.join(self.training_dir, 'config.json'))
        log_file = find_log_path(self.training_dir, train_config['vae_type'])

        batch_size = train_config['ds']['batch_size'] if 'batch_size' in train_config['ds'] else 64
        beta = train_config['beta']
        vae_type = train_config['vae_type']

        run_config = {'restore_model_path': last_model_path, 'log_file': log_file, 'batch_size': batch_size,
                      'beta': beta, 'vae_type': vae_type, 'out_dir': self.training_dir}
        sbatch_script_content = self.__prepare_sbatch_content(run_config)

        self.logger.info("Batch content: \n%s" % sbatch_script_content)
        sbatch_script_path = self.save_sbatch_script_to_file(sbatch_script_content)

        out_str = self.__read_sbatch_output(sbatch_script_path)
        jobid = self.__fetch_sbatch_jobid(out_str)
        self.logger.info("sbatch command output: %s, jobid fetched: %d" % (out_str, jobid))

        self.wait_until_state_is_running(jobid)

    def save_sbatch_script_to_file(self, script_content):
        existing_scripts = [p for p in os.listdir(self.out_sbatch_dir) if p.endswith(('sh',))]
        new_script_file_name = "%d.sh" % len(existing_scripts)

        new_script_path = op.join(self.out_sbatch_dir, new_script_file_name)
        with open(new_script_path, 'w') as f:
            f.write(script_content)

        self.logger.info("Written script content into path: %s" % new_script_path)
        return new_script_path

    def wait_until_state_is_running(self, jobid):
        squeue_out = self.__read_squeue_output()
        jobid_state = self.__fetch_state_for_jobid(squeue_out, jobid)

        while jobid_state == 'PD':
            self.logger.info("Checking current state for jobid: %d, state: %s" % (jobid, jobid_state))
            squeue_out = self.__read_squeue_output()
            jobid_state = self.__fetch_state_for_jobid(squeue_out, jobid)

            time.sleep(60)

        assert jobid_state == 'R'
        return jobid_state

    @staticmethod
    def __prepare_sbatch_content(run_config):
        header_templ = "#!/bin/bash -l\n#SBATCH -N 1\n#SBATCH -n 4\n#SBATCH --mem=10g\n#SBATCH --time 12:00:00\n" \
                       "#SBATCH -A grant\n#SBATCH -p partition\n" \
                       "#SBATCH --gres=gpu:1\n" \
                       "#SBATCH --output=\"%s\"\n#SBATCH " \
                       "--error=\"%s.err\""
        env_prepare_templ = "conda activate tf_gpu\nexport PYTHONPATH=`pwd`\n"
        invoke_script_cmd = "python src/models/vaes/scripts/train_vae.py --epochs_num 1000 --batch_size %s " \
                            "--restore_model_path %s %s %s %s" % (str(run_config['batch_size']),
                                                                  run_config['restore_model_path'],
                                                                  run_config['out_dir'], str(run_config['beta']),
                                                                  run_config['vae_type'])

        header = header_templ % (run_config['log_file'], run_config['log_file'])
        script_content = "%s\n\n\n%s\n\n" % (header, env_prepare_templ)
        script_content += invoke_script_cmd

        return script_content

    @staticmethod
    def __read_squeue_output():
        process = subprocess.Popen(['squeue'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate()
        return str(out, 'utf-8')

    @staticmethod
    def __read_sbatch_output(sbatch_script_path):
        process = subprocess.Popen(["sbatch", sbatch_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate()
        return str(out, 'utf-8')

    @staticmethod
    def __fetch_sbatch_jobid(sbatch_str):
        return int(sbatch_str.split(' ')[-1])

    @staticmethod
    def __fetch_state_for_jobid(squeue_str, jobid):
        lines = squeue_str.splitlines()
        jobs_hash = {}
        for l in lines[1:]:
            l_cleaned = [s for s in l.split(" ") if s]
            l_jobid = int(l_cleaned[0])
            jobs_hash[l_jobid] = l_cleaned[4]

        return jobs_hash[jobid]


def get_last_model_path_from_dir(training_dir):
    def get_iter_from_model_file(model_file):
        stripped_model_file = op.splitext(model_file)[0]
        iter_num = int(op.splitext(stripped_model_file)[0].split('-')[-1])
        return iter_num

    model_paths = [p for p in os.listdir(training_dir) if 'ckpt' in p]
    sorted_model_paths = sorted(model_paths, key=get_iter_from_model_file)

    return op.join(training_dir, op.splitext(sorted_model_paths[-1])[0])


@logger.log
def find_log_path(training_dir, vae_type):
    root_logs_dir = op.abspath(op.join(training_dir, '../../logs/%s_vae' % vae_type))
    logs_dir = op.join(root_logs_dir, op.basename(op.dirname(training_dir)))

    find_log_path.logger.info("Looking for log files in dir:  %s" % logs_dir)
    reg = "%s.*\\.txt" % op.basename(op.dirname(training_dir))

    potential_train_logs = [p for p in os.listdir(logs_dir) if re.match(reg, p)]
    base_log_file = "%s.txt" % op.basename(op.dirname(training_dir))

    if len(potential_train_logs) > 1:
        # assuming that log files end with _%d.txt
        curr_number = len(potential_train_logs)
        found_log_file = fs_utils.add_suffix_to_path(base_log_file, "%d" % curr_number)
    elif len(potential_train_logs) == 1:
        found_log_file = fs_utils.add_suffix_to_path(potential_train_logs[0], "1")
    else:
        found_log_file = base_log_file

    final_log_path = op.join(logs_dir, found_log_file)
    find_log_path.logger.info("Found new job log file: %s" % final_log_path)

    return final_log_path


@logger.log
def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    interval_in_secs = timedelta(seconds=12*60*60)

    job = PeriodicJob(interval_in_secs, args.existing_training_dir, args.out_sbatch_dir)
    job.start()

    while True:
        try:
            time.sleep(1)
        except ProgramKilled:
            job.join()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('existing_training_dir')
    parser.add_argument('out_sbatch_dir')
    return parser.parse_args()


if __name__ == '__main__':
    main()
