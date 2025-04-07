from typing import Any

import matplotlib.pyplot as plt

import lightning as lt


class XtVisualizerCallback(lt.Callback):
    def on_validation_start(self, trainer: lt.Trainer, pl_module: Any, tag: str = "training"):
        if not hasattr(pl_module, "dump"):
            return

        # numpyize dumped variables
        data_name: str = pl_module.dump.data_name
        logxt = pl_module.dump.xt.log10().mul_(10).cpu().numpy()
        _, M, _ = logxt.shape

        # plot xt
        fig, axs = plt.subplots(M, 1, sharex=True, figsize=[8, 1.5 * M])
        vmax = logxt.max()
        vmin = vmax - 80
        for m, ax in enumerate(axs):
            ax.imshow(logxt[..., m, :], origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

        fig.tight_layout(pad=0.1)
        pl_module.logger.experiment.add_figure(f"{tag}/{data_name}/xt", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def on_validation_end(self, trainer: lt.Trainer, pl_module: Any):
        self.on_validation_start(trainer, pl_module, tag="validation")
