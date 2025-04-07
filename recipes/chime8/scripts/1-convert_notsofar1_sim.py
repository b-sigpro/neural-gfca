#! /usr/bin/env python3

from argparse import ArgumentParser
from functools import partial
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf as oc
from progressbar import progressbar as pbar

import numpy as np

import soundfile as sf

sr = 16000


def convert_one(json_filename: Path, mode: str, dst_basepath: Path, duration: int, stepsize: int):
    basename = json_filename.stem
    info = oc.load(json_filename)

    # load mixture
    scale = float(info.columns.mixture_scale["values"])
    wav_mix = np.fromfile(json_filename.with_suffix(".mixture"), dtype=np.int16) / scale
    wav_mix = wav_mix.reshape(info.columns.mixture.shape).astype(np.float32)

    # load src
    scale = float(info.columns.gt_spk_direct_early_echoes_scale["values"])
    wav_src = np.fromfile(json_filename.with_suffix(".gt_spk_direct_early_echoes"), dtype=np.int16) / scale
    wav_src = wav_src.reshape(info.columns.gt_spk_direct_early_echoes.shape).astype(np.float32)

    # load activity
    act = np.fromfile(json_filename.with_suffix(".gt_spk_activity_scores"), dtype=np.int8) >= 0
    act = act.reshape(info.columns.gt_spk_activity_scores.shape)

    for tidx, t in enumerate(range(0, wav_mix.shape[0], stepsize * sr)):
        if (t_end := t + duration * sr) > wav_mix.shape[0]:
            break

        filename_prefix = f"{basename}-{tidx:05d}"

        act_ = act[t:t_end]
        spk_mask = act_.sum(0) > 0

        sf.write(dst_basepath / "mix" / f"{filename_prefix}.wav", wav_mix[t:t_end], sr, "PCM_24")
        for sidx, ss in enumerate(np.arange(wav_src.shape[-1])[spk_mask]):
            sf.write(dst_basepath / "src" / f"{filename_prefix}-s{sidx}.wav", wav_src[t:t_end, :, ss], sr, "PCM_24")
        np.save(dst_basepath / "act" / f"{filename_prefix}.npy", act_[:, spk_mask])


def convert(args, unk_args):
    from mpi4py.futures import MPIPoolExecutor

    with MPIPoolExecutor() as pool:
        for mode in ["train", "dev"] if args.mode is None else [args.mode]:
            print(f"mode: {mode}")

            dst_basepath = Path(f"./{mode}/") / "notsofar1-sim"
            (dst_basepath / "mix").mkdir(parents=True, exist_ok=True)
            (dst_basepath / "src").mkdir(parents=True, exist_ok=True)
            (dst_basepath / "act").mkdir(parents=True, exist_ok=True)

            ns_mode = "train" if mode == "train" else "val"
            json_filename_list = list(Path(f"./notsofar1-sim/{ns_mode}").glob("*.json"))

            convert_one_ = partial(
                convert_one, mode=mode, dst_basepath=dst_basepath, duration=args.duration, stepsize=args.stepsize
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
        filename_job = job_path / f"{mode}.sh"
        filename_stdout = out_path / f"{mode}.out"

        with open(filename_job, "w") as f:
            f.write(job_template)

            f.write("mpirun -np 320 -npernode 40 --hostfile $SGE_JOB_HOSTLIST ")
            f.write("-mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 ")
            f.write("-x MKL_NUM_THREADS=1 -x NUMEXPR_NUM_THREADS=1 -x OMP_NUM_THREADS=1 ")
            f.write(f"singularity exec --nv {dataset_path}/../singularity/singularity.sif direnv exec . ")
            f.write(f" python -m mpi4py.futures ./scripts/{command_name}.py gen --mode {mode} ")
            f.write(" ".join(unk_args) + "\n")

        os.system(f"qsub -g $JOB_GROUP $QSUB_ARGS -l rt_F=8 -l h_rt=12:0:0 -o {filename_stdout} {filename_job}")


def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("gen")
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
