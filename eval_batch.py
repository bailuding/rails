# Retrieval with Learned Similarities (RAILS).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Main entry point to run benchmarks.

import subprocess

algorithms = [
    "MoLBruteForceTopK",
    "MoLNaiveTopK5",
    "MoLNaiveFaissTopK5",
    "MoLNaiveTopK10",
    "MoLNaiveTopK50",
    "MoLNaiveTopK100",
    "MoLAvgTopK200",
    "MoLAvgTopK500",
    "MoLAvgTopK1000",
    #"MoLAvgTopK2500",
    #"MoLAvgTopK3000",
    "MoLAvgTopK2000",
    "MoLAvgTopK4000",
    "MoLCombTopK5_200",
    "MoLCombTopK50_500",
    #"MoLCombTopK50_1000",
    "MoLCombTopK100_1000",
]

configs = {
    "ml-1m": "configs/ml-1m/hstu-mol-sampled-softmax-n128-8x4x64-rails-final.gin",
    "ml-20m": "configs/ml-20m/hstu-mol-sampled-softmax-n128-8x4x128-rails-final.gin",
    "amzn-books": "configs/amzn-books/hstu-mol-sampled-softmax-n512-8x8x32-rails-final.gin",
}

checkpoints = {
    "ml-1m": "ckpts/ml-1m-l200/HSTU-b8-h2-dqk25-dv25-lsilud0.2-ad0.0_MoL-8x4x64-t0.05-d0.2-l2-q512d0.0swiglu-id0.1-gq128-gi128d0.0-gqi128d0.0-x-glu_silu-uids6040d0.5_local_ssl-n128-b128-lr0.001-wu0-wd0.001-2024-06-19_ep75",
    "ml-20m": "ckpts/ml-20m-l200/HSTU-b16-h8-dqk32-dv32-lsilud0.2-ad0.0_MoL-8x4x128-t0.05-d0.2-l2-q512d0.0swiglu-id0.1-gq128-gi128d0.0-gqi128d0.1-x-glu_silu-uids16384d0.8-l20.1_local_ssl-n128-ddp2avg-b64-lr0.001-wu0-wd0-2024-06-19_ep90",
    "amzn-books": "ckpts/amzn-books-l50/HSTU-b16-h8-dqk8-dv8-lsilud0.5-ad0.0_MoL-8x8x32-t0.05-d0.2-l2-q512d0.0geglu-id0.1-gq128-gi128d0.0-gqi128d0.0-x-glu_silu_local_ssl-n512-ddp2avg-b64-lr0.001-wu0-wd0-2024-06-20-fe5_ep115",
}

limit_eval_to_first_n = {
    "ml-1m": 0,
    "ml-20m": 0,
    "amzn-books": 8192,
}

def get_cmd(config_file, checkpoint, batch_size, algorithm, limit_eval_to_first_n):
    cmd = f"CUDA_VISIBLE_DEVICES=1 python3 eval_from_checkpoint.py --eval_batch_size={batch_size} --limit_eval_to_first_n={limit_eval_to_first_n} "
    cmd += f"--gin_config_file={config_file} --top_k_method={algorithm}  --inference_from_ckpt={checkpoint} --master_port=12346"
    return cmd

def run_eval(dataset, algorithm, batch_size):
    cmd = get_cmd(config_file = configs[dataset], checkpoint = checkpoints[dataset],
                    batch_size = batch_size, algorithm = algorithm, limit_eval_to_first_n = limit_eval_to_first_n[dataset])
    print(cmd)
    p = subprocess.Popen(cmd, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = p.communicate()
    result = None
    if p.returncode == 0:
        lines = output.splitlines()
        result = [lines[-2].decode('utf8').replace("INFO:root:", ""), lines[-1].decode('utf8').replace("INFO:root:", "")]
    else:
        print(p.returncode, output, error)
    return result
    
def eval(dataset, batch_size):
    eval_data = []
    for algorithm in algorithms:
        result = run_eval(dataset = dataset, algorithm = algorithm, batch_size = batch_size)
        if (len(eval_data)) == 0:
            eval_data.append("algorithm," + result[0])
        eval_data.append(algorithm + "," + result[1])
    return eval_data

if __name__ == "__main__":
    #dataset = "amzn-books"
    dataset = "ml-1m"
    #dataset = "ml-20m"
    batch_size = 32
    result = eval(dataset=dataset, batch_size=batch_size)
    print(f"================{dataset}===============")
    print("\n".join(result))