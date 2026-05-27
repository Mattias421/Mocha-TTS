from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import matcha.utils.monotonic_align as monotonic_align
from matcha import utils
from matcha.cli import MATCHA_URLS
from matcha.utils.model import generate_path, sequence_mask
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)


def _load_matcha_components(model: LightningModule, model_name: str = "matcha_ljspeech") -> str:
    save_dir = get_user_data_dir()
    ckpt_path = save_dir / f"{model_name}.ckpt"
    assert_model_downloaded(ckpt_path, MATCHA_URLS[model_name])

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    pretrained_state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for key, value in pretrained_state.items():
        if key.startswith("encoder.") or key.startswith("decoder."):
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                loaded += 1
            else:
                skipped += 1

    model.load_state_dict(model_state, strict=False)
    for param in model.encoder.parameters():
        param.requires_grad = False

    log.info(f"Loaded {loaded} encoder/decoder tensors from {ckpt_path}; skipped {skipped}.")
    return str(ckpt_path)


def _finetune_losses(model: LightningModule, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    x, x_lengths = batch["x"], batch["x_lengths"]
    y, y_lengths = batch["y"], batch["y_lengths"]
    spks = batch["spks"]

    with torch.no_grad():
        if model.n_spks > 1:
            spks = model.spk_emb(spks)

        mu_x, logw, x_mask = model.encoder(x, x_lengths, spks)
        y_mask = sequence_mask(y_lengths, y.shape[-1]).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if model.use_precomputed_durations:
            attn = generate_path(batch["durations"].squeeze(1), attn_mask.squeeze(1))
            logw_used = logw
        else:
            const = -0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=y.device)) * model.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const
            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1)).detach()
            logw_used = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask

        mu_x_for_alignment = mu_x
        if model.cde is not None:
            mu_x_for_alignment = model.cde(mu_x, x_mask, durations=torch.exp(logw_used).squeeze(1))

        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x_for_alignment.transpose(1, 2)).transpose(1, 2)

    diff_loss, _ = model.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=None)
    zero = torch.zeros((), device=diff_loss.device, dtype=diff_loss.dtype)
    return {"dur_loss": zero, "prior_loss": zero, "diff_loss": diff_loss}


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    matcha_name = cfg.get("pretrained_matcha_name", "matcha_ljspeech")
    ckpt = _load_matcha_components(model, matcha_name)
    log.info(f"Using pretrained Matcha checkpoint: {ckpt}")
    model.get_losses = lambda batch: _finetune_losses(model, batch)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting fine-tuning!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune_mocha.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)
    metric_dict, _ = train(cfg)
    return utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
