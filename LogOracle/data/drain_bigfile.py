import json
import time
from os.path import dirname
from tqdm import tqdm
from data.drain3 import TemplateMiner
from data.drain3.template_miner_config import TemplateMinerConfig


def drain3(ini_file, log_file, log_template_file, template_mined_file, my_logger):
    config = TemplateMinerConfig()
    config.load(ini_file)
    config.profiling_enabled = False
    template_miner = TemplateMiner(config=config)
    line_count = 0
    with open(log_file) as f:
        lines = f.readlines()
    start_time = time.time()
    line_to_cluster_id = []
    for line in tqdm(lines):
        result = template_miner.add_log_message(line)
        line_count += 1
        line_to_cluster_id.append(result["cluster_id"])
    time_took = time.time() - start_time
    rate = line_count / time_took
    with open(template_mined_file, 'w', encoding='utf-8') as f:
        for cluster_id in line_to_cluster_id:
            f.write(str(cluster_id-1) + '\n')  # cluster id starts from 1
    with open(log_template_file, 'w', encoding='utf-8') as f:
        for cluster in template_miner.drain.clusters:
            f.write(cluster.get_template() + '\n')
    my_logger.info(f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
                f"{len(template_miner.drain.clusters)} clusters")
    
