# SPDX-License-Identifier: MIT

import json
import os
from data.drain3.template_miner import TemplateMiner
from data.drain3.template_miner_config import TemplateMinerConfig
from data.drain3.file_persistence import FilePersistence
from tqdm import tqdm



def drain3(ini_file, log_file, log_template_file, template_mined_file, drain_path, args, my_logger):
    persistence_file = os.path.join(drain_path,'drain3_state.bin')
    persistence = FilePersistence(persistence_file)

    config = TemplateMinerConfig()
    config.load(ini_file)
    config.profiling_enabled = False

    template_miner = TemplateMiner(persistence, config)
    my_logger.info(f"Drain3 started with FILE persistence")
    my_logger.info(f"{len(config.masking_instructions)} masking instructions are in use")
    my_logger.info(f"Starting training mode. Reading from {log_file}")
    with open(log_file) as f:
        lines = f.readlines()
    line_to_cluster_id = []
    if args.phase == 'train':
        if os.path.exists(persistence_file):
            os.remove(persistence_file)
        for log_line in tqdm(lines):
            result = template_miner.add_log_message(log_line)
            result_json = json.dumps(result)
            line_to_cluster_id.append(result["cluster_id"])
        with open(template_mined_file, 'w', encoding='utf-8') as f:
            for cluster_id in line_to_cluster_id:
                f.write(str(cluster_id-1) + '\n')  # cluster id starts from 1
        with open(log_template_file, 'w', encoding='utf-8') as f:
            for cluster in template_miner.drain.clusters:
                f.write(cluster.get_template() + '\n')
        my_logger.info("Training done.")
    else:
        with open(log_template_file, 'a', encoding='utf-8') as f:
            for log_line in tqdm(lines):
                cluster = template_miner.match(log_line)
                if cluster is None:
                    my_logger.info(f"No match found, add a new template")
                    result = template_miner.add_log_message(log_line)
                    result_json = json.dumps(result)
                    template = result["template_mined"]
                    line_to_cluster_id.append(result["cluster_id"])
                    f.write(cluster.get_template() + '\n')
                else:
                    template = cluster.get_template()
                    line_to_cluster_id.append(cluster.cluster_id)
        with open(template_mined_file, 'w', encoding='utf-8') as f:
            for cluster_id in line_to_cluster_id:
                f.write(str(cluster_id-1) + '\n')  # cluster id starts from 1
        