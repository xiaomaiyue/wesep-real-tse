# Copyright (c) 2020 Mobvoi Inc (Di Wu)
#               2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import os.path
import re

import torch


def get_args():
    parser = argparse.ArgumentParser(description="average model")
    parser.add_argument("--dst_model", required=True, help="averaged model")
    parser.add_argument("--src_path",
                        required=True,
                        help="src model path for average")
    parser.add_argument("--num",
                        default=5,
                        type=int,
                        help="nums for averaged model")
    parser.add_argument(
        "--min_epoch",
        default=0,
        type=int,
        help="min epoch used for averaging model",
    )
    parser.add_argument(
        "--max_epoch",
        default=65536,  # Big enough
        type=int,
        help="max epoch used for averaging model",
    )
    parser.add_argument(
        "--mode",
        default="final",
        choices=["final", "best"],
        type=str,
        help="final: average the last --num checkpoints; best: average --epochs",
    )
    parser.add_argument(
        "--epochs",
        default="",
        type=str,
        help="comma-separated checkpoint epochs used by --mode best",
    )
    args = parser.parse_args()
    print(args)
    return args


def checkpoint_epoch(path):
    match = re.search(r"checkpoint_(\d+)\.pt$", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))


def find_checkpoints(src_path, min_epoch, max_epoch):
    paths = []
    for path in glob.glob(os.path.join(src_path, "checkpoint_*.pt")):
        epoch = checkpoint_epoch(path)
        if epoch is None:
            continue
        if min_epoch <= epoch <= max_epoch:
            paths.append((epoch, path))
    return [path for _, path in sorted(paths)]


def select_checkpoints(args):
    if args.mode == "final":
        path_list = find_checkpoints(args.src_path, args.min_epoch,
                                     args.max_epoch)
        if len(path_list) < args.num:
            raise RuntimeError(
                f"Need {args.num} checkpoints, but found {len(path_list)} "
                f"in {args.src_path} within epoch range "
                f"[{args.min_epoch}, {args.max_epoch}].")
        return path_list[-args.num:]

    if not args.epochs.strip():
        raise ValueError("--epochs must be set when --mode best is used.")

    epoch_indexes = [x.strip() for x in args.epochs.split(",") if x.strip()]
    if len(epoch_indexes) != args.num:
        raise ValueError(
            f"--num is {args.num}, but --epochs contains "
            f"{len(epoch_indexes)} epochs: {args.epochs}")

    path_list = [
        os.path.join(args.src_path, "checkpoint_" + x + ".pt")
        for x in epoch_indexes
    ]
    missing = [p for p in path_list if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    return path_list


def main():
    args = get_args()
    path_list = select_checkpoints(args)
    print(path_list)
    avg = None
    num = args.num
    for path in path_list:
        print("Processing {}".format(path))
        states = torch.load(path, map_location=torch.device("cpu"))
        states = states["models"][0] if "models" in states else states
        if avg is None:
            avg = {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in states.items()
            }
        else:
            for k in avg.keys():
                if torch.is_tensor(avg[k]) and torch.is_floating_point(avg[k]):
                    avg[k] += states[k]
                else:
                    avg[k] = states[k]
    # average
    for k in avg.keys():
        if torch.is_tensor(avg[k]) and torch.is_floating_point(avg[k]):
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    avg = {"models": [avg]}
    print("Saving to {}".format(args.dst_model))
    torch.save(avg, args.dst_model)


if __name__ == "__main__":
    main()
