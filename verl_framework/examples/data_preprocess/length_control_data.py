# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import datasets



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/map-vepfs/zhangwei/LengthFollowing-data/processed_data_4")

    args = parser.parse_args()

    data_source = 'length_control'

    # 本地目录
    local_dir = args.local_dir
    # dataset = datasets.load_dataset(data_source, "main")
    dataset = datasets.load_dataset("json", data_files={"train": os.path.join(local_dir, "train.jsonl"),
                                                        "test": os.path.join(local_dir, "valid.jsonl")})

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("prompt")

            question = question_raw + ' /no_think'
            answer_raw = example.pop("word_count")
            solution = answer_raw
            raw_id = example.pop("id")
            raw_lang = example.pop("lang")
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "target_len": solution,
                "reward_model": {"style": "rule", "ground_truth": solution, "split":split},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "id":raw_id,
                    "lang":raw_lang
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

