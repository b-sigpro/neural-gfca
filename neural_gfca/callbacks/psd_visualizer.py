from typing import Any

import matplotlib.pyplot as plt

import lightning as lt


class PsdVisualizerCallback(lt.Callback):
    def on_validation_start(self, trainer: lt.Trainer, pl_module: Any, tag: str = "training"):
        if not hasattr(pl_module, "dump"):
            return

        # numpyize dumped variables
        data_name: str = pl_module.dump.data_name
        logx = pl_module.dump.x.abs().log10_().mul_(20).cpu().numpy()
        loglm = pl_module.dump.lm.log10().mul_(10).cpu().numpy()
        z = pl_module.dump.z.cpu().numpy()
        w = pl_module.dump.w.cpu().numpy()
        act = pl_module.dump.act.cpu().numpy()

        F, N, T = loglm.shape

        # plot observation and PSDs
        gridspec_kw = dict(height_ratios=[2] + N * [2, 0.5, 0.5])
        fig, axs = plt.subplots(1 + (3 * N), 1, sharex=True, gridspec_kw=gridspec_kw, figsize=[8, 2 + 2 * N])

        vmax = logx.max()
        axs[0].imshow(logx, origin="lower", aspect="auto", vmax=vmax, vmin=vmax - 80)

        lmmax = loglm.max()
        lmmin = lmmax - 80
        zmin, zmax = z.min(), z.max()
        for n, (ax1, ax2, ax3) in enumerate(axs[1 : 1 + 3 * N].reshape(-1, 3)):
            ax1.imshow(loglm[..., n, :], origin="lower", aspect="auto", vmin=lmmin, vmax=lmmax)

            ax2.plot(act[n])
            ax2.plot(w[..., n, :].T)
            ax2.set_xlim(0, T - 1)
            ax2.set_ylim(-0.1, 1.1)

            ax3.plot(z[..., n, :].T)
            ax3.set_xlim(0, T - 1)
            ax3.set_ylim(zmin, zmax)

        fig.tight_layout(pad=0.1)
        pl_module.logger.experiment.add_figure(f"{tag}/{data_name}/psd", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def on_validation_end(self, trainer: lt.Trainer, pl_module: Any):
        self.on_validation_start(trainer, pl_module, tag="validation")
