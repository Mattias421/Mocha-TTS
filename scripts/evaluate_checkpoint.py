#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import soundfile as sf
import torch

from matcha.cli import (
    MATCHA_URLS,
    SINGLESPEAKER_MODEL,
    VOCODER_URLS,
    load_matcha,
    load_vocoder,
    process_text,
    to_waveform,
)
from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wavs and evaluate MCD/F0 from a Matcha checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to custom model checkpoint.")
    parser.add_argument(
        "--use_official_ckpt",
        action="store_true",
        help="Use official Matcha checkpoint (default model: matcha_ljspeech).",
    )
    parser.add_argument("--model_name", type=str, default="matcha_ljspeech", choices=MATCHA_URLS.keys())
    parser.add_argument("--vocoder", type=str, default=None, choices=VOCODER_URLS.keys())
    parser.add_argument("--filelist", type=str, required=True, help="Path to filelist like wav|text.")
    parser.add_argument("--outdir", type=str, required=True, help="Directory for generated wavs + metrics.")
    parser.add_argument("--max_utts", type=int, default=None, help="Optional cap on utterances.")
    parser.add_argument("--steps", type=int, default=10, help="Flow matching decode steps.")
    parser.add_argument("--temperature", type=float, default=0.667)
    parser.add_argument("--length_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--f0_nj", type=int, default=8)
    parser.add_argument("--mcd_nj", type=int, default=8)
    return parser.parse_args()


def resolve_model_ckpt(args: argparse.Namespace) -> tuple[str, str]:
    if args.use_official_ckpt:
        save_dir = get_user_data_dir()
        ckpt_path = save_dir / f"{args.model_name}.ckpt"
        assert_model_downloaded(ckpt_path, MATCHA_URLS[args.model_name])
        return args.model_name, str(ckpt_path)

    if args.checkpoint_path is None:
        raise ValueError("Either --checkpoint_path or --use_official_ckpt must be provided.")
    ckpt_path = resolve_checkpoint_path(Path(args.checkpoint_path))
    return "custom_model", str(ckpt_path)


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [path / "checkpoints" / "last.ckpt", path / "last.ckpt"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        ckpt_glob = sorted((path / "checkpoints").glob("checkpoint_*.ckpt"))
        if ckpt_glob:
            return ckpt_glob[-1]
    raise FileNotFoundError(f"Could not resolve checkpoint file from: {path}")


def _remap_legacy_cde_keys(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    remapped = dict(state_dict)
    mapping = {
        "cde.func.linear1.weight": "cde.func.hidden_layers.0.weight",
        "cde.func.linear1.bias": "cde.func.hidden_layers.0.bias",
        "cde.func.linear2.weight": "cde.func.out.weight",
        "cde.func.linear2.bias": "cde.func.out.bias",
    }
    n = 0
    for src, dst in mapping.items():
        if src in remapped and dst not in remapped:
            remapped[dst] = remapped.pop(src)
            n += 1
    return remapped, n


def load_custom_matcha_with_fallback(ckpt_path: str, device: torch.device):
    try:
        return load_matcha("custom_model", ckpt_path, device)
    except RuntimeError as exc:
        if "cde.func.hidden_layers.0.weight" not in str(exc):
            raise

    print("[!] Detected legacy CDE checkpoint format. Applying key remap fallback...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters")
    if hparams is None:
        raise ValueError("Checkpoint is missing 'hyper_parameters'; cannot construct MatchaTTS for fallback load.")

    model = MatchaTTS(**hparams)
    state_dict = checkpoint["state_dict"]
    remapped_state_dict, n_remapped = _remap_legacy_cde_keys(state_dict)
    print(f"[!] Remapped {n_remapped} legacy CDE parameter keys.")
    model.load_state_dict(remapped_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print("[+] custom_model loaded with legacy CDE fallback!")
    return model


def resolve_vocoder(args: argparse.Namespace, model_name: str) -> tuple[str, str]:
    vocoder_name = args.vocoder
    if vocoder_name is None:
        if model_name in SINGLESPEAKER_MODEL:
            vocoder_name = SINGLESPEAKER_MODEL[model_name]["vocoder"]
        else:
            vocoder_name = "hifigan_univ_v1"

    save_dir = get_user_data_dir()
    vocoder_path = save_dir / vocoder_name
    assert_model_downloaded(vocoder_path, VOCODER_URLS[vocoder_name])
    return vocoder_name, str(vocoder_path)


def load_filelist(path: Path, max_utts: int | None) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        wav_path, text = line.split("|", 1)
        utt_id = Path(wav_path).stem
        entries.append((utt_id, wav_path, text))
        if max_utts is not None and len(entries) >= max_utts:
            break
    return entries


@torch.inference_mode()
def generate_wavs(
    entries: list[tuple[str, str, str]],
    model,
    vocoder,
    denoiser,
    out_wav_dir: Path,
    device: torch.device,
    steps: int,
    temperature: float,
    length_scale: float,
):
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    for i, (utt_id, _, text) in enumerate(entries, start=1):
        text_info = process_text(i, text, device)
        output = model.synthesise(
            text_info["x"],
            text_info["x_lengths"],
            n_timesteps=steps,
            temperature=temperature,
            spks=None,
            length_scale=length_scale,
        )
        waveform = to_waveform(output["mel"], vocoder, denoiser)
        sf.write(out_wav_dir / f"{utt_id}.wav", waveform.cpu().numpy(), 22050, "PCM_24")


def run_metric_script(script: Path, gen_dir: Path, gt_dir: Path, out_dir: Path, nj: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/python",
        str(script),
        str(gen_dir),
        str(gt_dir),
        "--outdir",
        str(out_dir),
        "--nj",
        str(nj),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Metric script failed: {' '.join(cmd)}. "
            "Ensure evaluation deps are installed (e.g. pysptk, pyworld, fastdtw)."
        ) from exc


def run_mcd_v2_script(script: Path, gen_dir: Path, gt_dir: Path):
    cmd = [
        ".venv/bin/python",
        str(script),
        str(gt_dir),
        str(gen_dir),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Metric script failed: {' '.join(cmd)}. "
            "Ensure evaluation deps are installed (e.g. pysptk, pyworld, fastdtw)."
        ) from exc


def parse_avg_result(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model_name, ckpt_path = resolve_model_ckpt(args)
    vocoder_name, vocoder_path = resolve_vocoder(args, model_name)

    outdir = Path(args.outdir)
    gen_dir = outdir / "generated_wavs"
    metrics_dir = outdir / "metrics"
    mcd_out = metrics_dir / "mcd"
    f0_out = metrics_dir / "f0"

    entries = load_filelist(Path(args.filelist), args.max_utts)
    if len(entries) == 0:
        raise ValueError("No utterances found in filelist.")

    if model_name == "custom_model":
        model = load_custom_matcha_with_fallback(ckpt_path, device)
    else:
        model = load_matcha(model_name, ckpt_path, device)
    vocoder, denoiser = load_vocoder(vocoder_name, vocoder_path, device)

    generate_wavs(
        entries=entries,
        model=model,
        vocoder=vocoder,
        denoiser=denoiser,
        out_wav_dir=gen_dir,
        device=device,
        steps=args.steps,
        temperature=args.temperature,
        length_scale=args.length_scale,
    )

    gt_dir = outdir / "gt_wavs"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for utt_id, wav_path, _ in entries:
        target = gt_dir / f"{utt_id}.wav"
        if not target.exists():
            target.symlink_to(Path(wav_path).resolve())

    mcd_out.mkdir(parents=True, exist_ok=True)
    run_mcd_v2_script(Path("scripts/evaluate_mcd_v2.py"), gen_dir, gt_dir)
    run_metric_script(Path("scripts/evaluate_f0.py"), gen_dir, gt_dir, f0_out, args.f0_nj)

    summary = {
        "checkpoint": ckpt_path,
        "model_name": model_name,
        "vocoder": vocoder_name,
        "device": str(device),
        "num_utts": len(entries),
        "steps": args.steps,
        "mcd": parse_avg_result(gen_dir / "evaluation_results.txt"),
        "log_f0_rmse": parse_avg_result(f0_out / "log_f0_rmse_avg_result.txt"),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
