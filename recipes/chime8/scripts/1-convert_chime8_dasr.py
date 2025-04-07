#! /usr/bin/env python3

from argparse import ArgumentParser
from functools import partial
import json
from math import ceil, floor
import os
from pathlib import Path
import sys

from progressbar import progressbar as pbar

import numpy as np

import soundfile as sf

sr = 16000


def convert_one(json_filename: Path, data: str, wav_path: Path, dst_basepath: Path, duration: int, stepsize: int):
    basename = json_filename.stem

    # load mixture
    wav_filename_list = list(wav_path.glob(f"{basename}*CH*.wav")) + list(wav_path.glob(f"{basename}*CH*.flac"))
    wav_filename_list.sort()

    if data == "mixer6":
        wav_filename_list = filter(lambda x: str(x)[-9:-5] not in ["CH01", "CH02", "CH03"], wav_filename_list)

    wav_mix_list = [sf.read(wav_filename, dtype="float32")[0] for wav_filename in wav_filename_list]
    min_len = min(wav.shape[0] for wav in wav_mix_list)

    wav_mix = np.stack([wav[:min_len] for wav in wav_mix_list], axis=-1)

    # build activity
    with open(json_filename) as f:
        transcriptions = json.load(f)

    speaker_list = list({utt["speaker"] for utt in transcriptions})
    act = np.zeros([wav_mix.shape[0], len(speaker_list)], dtype=bool)
    for utt in transcriptions:
        uidx = speaker_list.index(utt["speaker"])

        start_time = floor(float(utt["start_time"]) * sr)
        end_time = ceil(float(utt["end_time"]) * sr)

        act[start_time:end_time, uidx] = True

    for tidx, t in enumerate(range(0, wav_mix.shape[0], stepsize * sr)):
        if (t_end := t + duration * sr) > wav_mix.shape[0]:
            break

        filename_prefix = f"{basename}-{tidx:05d}"

        act_ = act[t:t_end]
        spk_mask = act_.sum(0) > 0

        sf.write(dst_basepath / "mix" / f"{filename_prefix}.wav", wav_mix[t:t_end], sr, "PCM_24")
        np.save(dst_basepath / "act" / f"{filename_prefix}.npy", act[t:t_end, spk_mask])


def convert(args, unk_args):
    from mpi4py.futures import MPIPoolExecutor

    with MPIPoolExecutor() as pool:
        for mode in ["train", "dev"] if args.mode is None else [args.mode]:
            print(f"mode: {mode}")

            dst_basepath = Path(f"./{mode}/") / args.data
            (dst_basepath / "mix").mkdir(parents=True, exist_ok=True)
            (dst_basepath / "act").mkdir(parents=True, exist_ok=True)

            json_filename_list = list((Path("./chime8_dasr/") / args.data / "transcriptions" / mode).glob("*.json"))
            wav_path = Path("./chime8_dasr/") / args.data / "audio" / mode

            convert_one_ = partial(
                convert_one,
                data=args.data,
                wav_path=wav_path,
                dst_basepath=dst_basepath,
                duration=args.duration,
                stepsize=args.stepsize,
            )
            for _ in pbar(pool.map(convert_one_, json_filename_list)):
                pass


def submit_jobs(args, unk_args):
    assert sys.version_info >= (3, 7)

    script_path = Path(__file__)
    dataset_path = script_path.parent.parent
    command_name = script_path.stem

    job_path = Path(f"jobs/{command_name}/")
    out_path = Path(f"jobs.out/{command_name}/")
    job_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(dataset_path / "scripts" / "job_template.sh") as f:
        job_template = f.read()

    for mode in ["train", "dev"]:
        for data in ["chime6", "dipco", "mixer6", "notsofar1"]:
            filename_job = job_path / f"{data}-{mode}.sh"
            filename_stdout = out_path / f"{data}-{mode}.out"

            with open(filename_job, "w") as f:
                f.write(job_template)

                f.write("mpirun -np 80 -npernode 10 --hostfile $SGE_JOB_HOSTLIST ")
                f.write("-mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 ")
                f.write("-x MKL_NUM_THREADS=1 -x NUMEXPR_NUM_THREADS=1 -x OMP_NUM_THREADS=1 ")
                f.write(f"singularity exec --nv {dataset_path}/../singularity/singularity.sif direnv exec . ")
                f.write(f" python -m mpi4py.futures ./scripts/{command_name}.py gen {data} --mode {mode} ")
                f.write(" ".join(unk_args) + "\n")

            os.system(f"qsub -g $JOB_GROUP $QSUB_ARGS -l rt_F=8 -l h_rt=12:0:0 -o {filename_stdout} {filename_job}")


def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("gen")
    sub_parser.add_argument("data", type=str)
    sub_parser.add_argument("--mode", type=str, default="train")
    sub_parser.add_argument("--duration", type=int, default=30)
    sub_parser.add_argument("--stepsize", type=int, default=15)
    sub_parser.set_defaults(handler=convert)

    sub_parser = sub_parsers.add_parser("sub")
    sub_parser.set_defaults(handler=submit_jobs)

    args, unk_args = parser.parse_known_args()
    if hasattr(args, "handler"):
        args.handler(args, unk_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
