#! /usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path

from progressbar import progressbar as pbar

import torch

import cupy as cp
from wpe import wpe

from librosa.core import istft, stft
import soundfile as sf


def dereverberate_mixtures(args):
    """
    Note that we used our torch implementation of WPE in our ICASSP paper,
    while here we replaced it with the more standard `gpu-wpe`.
    If there is a reproduction issue, please let us know.
    """

    mode_list = ["train", "dev"] if args.mode is None else [args.mode]

    for mode in mode_list:
        base_path = Path(f"./{mode}") / args.data

        dst_path = base_path / "derev"
        dst_path.mkdir(exist_ok=True)

        src_filename_list = sorted((base_path / "mix").glob("*.wav"))
        if os.environ["SGE_TASK_ID"] != "undefined":
            start = int(os.environ["SGE_TASK_ID"]) - 1
            end = start + int(os.environ["SGE_TASK_STEPSIZE"])

            src_filename_list = src_filename_list[start:end]

        for src_filename in pbar(src_filename_list, redirect_stdout=True):  # type: ignore
            src_filename: Path

            src_wav, sr = sf.read(src_filename)

            src_spec = stft(src_wav.T, n_fft=1024, hop_length=256)  # [M, F, T]
            src_spec = torch.as_tensor(src_spec, device="cuda")
            M, F, T = src_spec.shape

            if src_spec.abs().square().amax(dim=1).amin() == 0:
                continue

            if src_spec.abs().square().mean() < 1e-6:
                print("Warning almost zero!", src_filename)
                continue

            dst_spec = torch.from_dlpack(wpe(cp.from_dlpack(src_spec).transpose(1, 0, 2), taps=5, delay=2))

            dst_wav = istft(dst_spec.cpu().numpy(), hop_length=256).T
            sf.write(dst_path / src_filename.name, dst_wav, sr, "PCM_24")


def submit_jobs(args):
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent
    command_name = script_path.stem

    singularity_path = dataset_path.parent / "singularity" / "singularity.sif"

    job_path = Path(f"jobs/{command_name}/")
    out_path = Path(f"jobs.out/{command_name}/")
    job_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(f"{dataset_path}/scripts/job_template.sh") as f:
        job_template = f.read()

    for mode in ["train", "dev"]:
        for data in ["chime6", "dipco", "mixer6", "notsofar1", "notsofar1-sim"]:
            fname_job = job_path / f"{mode}-{data}.sh"

            n_mix = len(list((Path(mode) / data / "mix").glob("*.wav")))

            with open(fname_job, "w") as f:
                f.write(job_template)
                f.write(
                    f"""
#$-o jobs.out/{command_name}/{mode}-{data}-$TASK_ID.txt"
LOCAL_STEPSIZE=$((SGE_TASK_STEPSIZE / 4))
for gpu in 0 1 2 3; do
    singularity exec --nv \\
        --env CUDA_VISIBLE_DEVICES=$gpu \\
        --env SGE_TASK_ID=$((SGE_TASK_ID + LOCAL_STEPSIZE * gpu)) \\
        --env SGE_TASK_STEPSIZE=$LOCAL_STEPSIZE \\
        {singularity_path} direnv exec . \\
        python ./scripts/{command_name}.py drv {data} --mode {mode} \\
        > jobs.out/{command_name}/{mode}-{data}-$SGE_TASK_ID-$gpu.txt 2>&1 &
done
wait    
    """
                )

            os.system(f"qsub -g $JOB_GROUP -l rt_F=1 -l h_rt=2:0:0 -t 1-{n_mix}:{args.n_mix_per_job} {fname_job}")


if __name__ == "__main__":
    # argument parser
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    # generate_mixtures
    sub_parser = sub_parsers.add_parser("drv", help="dereverberate mixture signals")
    sub_parser.add_argument("data", type=str)
    sub_parser.add_argument("--mode", type=str, default=None)
    sub_parser.set_defaults(handler=dereverberate_mixtures)

    # submit_jobs
    sub_parser = sub_parsers.add_parser("sub", help="submit PBS jobs")
    sub_parser.add_argument("--n_mix_per_job", type=int, default=1000)
    sub_parser.set_defaults(handler=submit_jobs)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
