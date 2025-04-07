#!/usr/bin/env python3

from argparse import ArgumentParser
from math import ceil
import os
from pathlib import Path
import pickle as pkl

from progressbar import ProgressBar

import numpy as np

import soundfile as sf


class EmptySample(Exception):
    pass


def make_dataset(args, unk_args):
    import h5py
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("================================")
        print("Parameters")
        print("--------------------------------")
        for key, val in args.__dict__.items():
            print(f"{key:20s}: {val}")
        print("================================")

    mix_path = Path.cwd() / args.mode / args.data / "derev"
    act_path = Path.cwd() / args.mode / args.data / "act"

    # obtain file list
    if rank == 0:
        wav_filename_list = sorted(mix_path.glob("*.wav"))

        n_fname = len(wav_filename_list)
        wav_filename_list += (ceil(n_fname / size) * size - n_fname) * [None]

        group_list = []
    else:
        wav_filename_list = None
    wav_filename_list = comm.bcast(wav_filename_list, root=0)

    hdf_name = Path("./hdf5") / f"unsupervised.{args.data}-nsrc{args.n_src}-{args.mode}.hdf5"
    with h5py.File(hdf_name, "w", driver="mpio", comm=comm) as f:
        pbar = ProgressBar(redirect_stdout=True) if rank == 0 else lambda x: x
        for widx, wav_filename in enumerate(pbar(wav_filename_list[rank::size])):
            # load spectrogram
            try:
                if wav_filename is None:
                    raise EmptySample()

                wav_mix, sr = sf.read(wav_filename, dtype="float32")

                duration, n_mic = wav_mix.shape
                if duration < sr * args.duration:
                    raise EmptySample()

                act = np.load(act_path / f"{wav_filename.stem}.npy")
                _, n_src = act.shape

                if n_src > args.n_src or act.sum() == 0:
                    raise EmptySample()

                grp_name = f"{size * widx + rank:08d}"
            except EmptySample:
                grp_name, duration, n_mic, n_src = None, None, None, None
            except Exception as e:
                print(e)
                grp_name, duration, n_mic, n_src = None, None, None, None

            # initialize datasets
            all_grp_names = filter(None, comm.allgather(grp_name))
            all_durations = filter(None, comm.allgather(duration))
            all_n_mics = filter(None, comm.allgather(n_mic))
            all_n_srcs = filter(None, comm.allgather(n_src))
            for grp_name_, duration_, n_mic_, n_src_ in zip(
                all_grp_names,
                all_durations,
                all_n_mics,
                all_n_srcs,
                strict=False,
            ):
                g = f.create_group(grp_name_)
                g.create_dataset("wav", [n_mic_, duration_], "float32")
                g.create_dataset("act", [n_src_, duration_], "bool")

                if rank == 0:
                    group_list.append(grp_name_)

            # store data
            if grp_name is not None:
                f[f"{grp_name}/wav"][:] = wav_mix.T
                f[f"{grp_name}/act"][:] = act.T

    if rank == 0:
        with open(hdf_name.with_suffix(".pkl"), "wb") as f:
            pkl.dump(group_list, f)


def submit_jobs(args, unk_args):
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent
    command_name = script_path.stem

    job_path = Path(f"jobs/{command_name}/")
    out_path = Path(f"jobs.out/{command_name}/")
    job_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(f"{dataset_path}/scripts/job_template.sh") as f:
        job_template = f.read()

    for mode in ["train", "dev"]:
        for data in ["chime6", "dipco", "mixer6", "notsofar1"]:
            fname_job = job_path / f"{mode}-{data}.sh"
            fname_stdout = out_path / f"{mode}-{data}.out"

            with open(fname_job, "w") as f:
                f.write(job_template)

                f.write("mpirun -np 320 -npernode 40 --hostfile $SGE_JOB_HOSTLIST ")
                f.write("--mca btl ^openib --mca btl_tcp_if_include eno1 ")
                f.write(f"singularity exec --nv {dataset_path}/../singularity/singularity.sif direnv exec . ")
                f.write(f" python ./scripts/{command_name}.py gen {data} --mode {mode} ")
                f.write(" ".join(unk_args) + "\n")

            os.system(f"qsub -g $JOB_GROUP $QSUB_ARGS -l rt_F=8 -l h_rt=1:0:0 -o {fname_stdout} {fname_job}")


def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("gen", help="dereverberate mixture signals")
    sub_parser.add_argument("data", type=str)
    sub_parser.add_argument("--mode", type=str, default="tr")
    sub_parser.add_argument("--n_src", type=int, default=4)
    sub_parser.add_argument("--duration", type=int, default=4)
    sub_parser.set_defaults(handler=make_dataset)

    sub_parser = sub_parsers.add_parser("sub", help="dereverberate mixture signals")
    sub_parser.set_defaults(handler=submit_jobs)

    args, unk_args = parser.parse_known_args()
    if hasattr(args, "handler"):
        args.handler(args, unk_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
