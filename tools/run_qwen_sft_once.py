import argparse
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/public_checkpoints/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Directory that contains the checkpoint to load or a folder of checkpoints",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint path to load. Use 'last' to pick the newest checkpoint under --model_dir",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device identifier passed to device_map for model loading",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Floating point precision for model weights",
    )
    return parser.parse_args()


def find_latest_checkpoint(model_dir: str) -> Optional[str]:
    """Return the latest checkpoint directory under model_dir if any."""
    root = Path(model_dir)
    if not root.exists():
        return None

    candidates = []
    for path in root.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        candidates.append((step, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return str(candidates[-1][1])


def resolve_model_path(model_dir: str, resume_from: Optional[str]) -> str:
    if resume_from is None:
        return model_dir
    if resume_from == "last":
        latest = find_latest_checkpoint(model_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found under {model_dir}")
        return latest
    return resume_from


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model_dir, args.resume_from)

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    model_dir_path = Path(model_path)
    hf_model_path = model_dir_path
    tokenizer_path = model_dir_path

    # RL checkpoints store Hugging Face model under `llm/` and tokenizer under `tokenizer/`
    if (model_dir_path / "llm").is_dir():
        hf_model_path = model_dir_path / "llm"
    if (model_dir_path / "tokenizer").exists():
        tokenizer_path = model_dir_path / "tokenizer"

    if not (hf_model_path / "config.json").exists():
        raise FileNotFoundError(
            f"Cannot locate config.json under {hf_model_path}. "
            "Make sure --model_dir or --resume_from points to a HuggingFace-formatted folder or RL checkpoint."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_model_path),
        device_map=args.device,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 简单生成
    prompt = f"""
        You are a recommendation system for Musical products.\n\nUser's purchase history: Fender FB620 Electric Bass Gig Bag, Black\n\nPlease rank the following 20 items by likelihood of purchase.\n\nCandidate items:\n1. John Pearse Acoustic Strings Phosphor Bronze Light 12-53 (3 Pack Bundle)\n2. Guitar Neck Notched Straight Edge Fret Rocker String Height Gauge, Luthiers Tool for Guitar Fretboard and Frets\n3. Neewer 3 Pin XLR Solder Type Connector (20 pcs-black)\n4. Green Paua Abalone Inlay Dots 6mm Guitar Ukulele Banjo Maker (Pack of 20)\n5. StewMac Fret Leveling File, 6\" Length\n6. MXR DC Brick Power Supply\n7. GHS Strings BB40M Bright Bronze, 80/20 Copper-Zinc Alloy, Acoustic Guitar Strings, Medium (.013-.056)\n8. Pirastro Gold Rosin For Violin - Viola - Cello\n9. Rockville Dual UHF 15 Channel Metal Handheld Wireless Microphone System (RWM65U), Black\n10. Jim Dunlop Body & Fingerboard Cleaning Kit (6503)\n11. Vandoren SRMIXT35 Tenor Sax Jazz Reed Mix Card includes 1 each ZZ, V16, JAVA and JAVA Red Strength 3.5\n12. Audio-Technica AT2020 Condenser Studio Microphone Bundle with Studio Stand, Pop Filter and XLR Cable (4 Items)\n13. Standard 4 Bolt\"Custom Built\" Engraved or Printed Guitar Neck Plate - Choose from 4 designs - Silver, Gold or Black\n14. lotmusic A0027 6pcs Chrome Guitar String Tuning Pegs Tuners Machine Heads Guitar Parts\n15. Ukulele Hard Case, Tenor Ukulele Case, 26 Inch Crocodile Pattern Leather Bulge Surface with Plush Interior Wooden Case (Black)\n16. Hola! HM-MS+ Professional Folding Orchestra Sheet Music Stand + Carry Bag\n17. Little Wedgie Guitar Amplifier Tilting and Isolation Wedge\n18. Seymour Duncan SH-5 Duncan Custom Humbucker Pickup - Black\n19. BOSS Roland PSA-120S\n20. Wayfinder Supply Co. Lightweight Electric Guitar Gig Bag (WF-GB-ELEC),Grey\n\nIMPORTANT: Your response must end with exactly one line in this format:\nRANKING: number1,number2,number3,number4,number5,number6,number7,number8,number9,number10,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20\n\nWhere each number is between 1-20. Example:\nRANKING: 26,45,78,50,38,99,77,43,53,89,8,93,97,52,47,31,48,83,98,79\n
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(input_ids, max_new_tokens=9000)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
