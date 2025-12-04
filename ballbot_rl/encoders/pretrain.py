import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
import pickle
import argparse
from termcolor import colored

from ballbot_rl.data import DepthImageDataset, collect_depth_image_paths, load_depth_images
from ballbot_rl.encoders import TinyAutoencoder, train_autoencoder


def main(args):

    if args.data_path[-4:] == ".pkl":
        print("loading from pickled dataset...")
        data_path = Path(args.data_path).resolve()
        with data_path.open("rb") as fl:
            dataset = pickle.load(fl)
    else:
        print("reading from raw dataset, this can take a while...")

        root_directory = Path(args.data_path).resolve()
        image_paths = collect_depth_image_paths(root_directory)
        image_data = load_depth_images(image_paths)

        dataset = DepthImageDataset(image_data)

    print(colored(f"dataset contains {len(dataset)} depth images"))

    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=True,
        num_workers=4)  #I'm also shuffling here for display reasons

    if args.save_dataset_as_pickle:
        save_path = Path(args.save_dataset_as_pickle).resolve()
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / "dataset.pkl").open("wb") as fl:
            pickle.dump(dataset, fl)

    if not args.from_model:
        model = TinyAutoencoder(H=64, W=64)
    else:
        model_path = Path(args.from_model).resolve()
        model = torch.load(str(model_path), weights_only=False)

    save_encoder_path = Path(args.save_encoder_to).resolve()
    train_autoencoder(model,
                      train_loader,
                      val_loader,
                      device=torch.device("cuda"),
                      epochs=100,
                      lr=1e-3,
                      save_path=str(save_encoder_path))


def cli_main():
    """CLI entry point for encoder pretraining."""
    _parser = argparse.ArgumentParser(description="Pretrain small encoder")
    _parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help=
        "Either a pkl file, or a directory that has the structure from the gather_data.py logging script"
    )
    _parser.add_argument(
        "--save_dataset_as_pickle",
        type=str,
        default="",
        help="an optional path where the dataset will be saved as a pkl")
    _parser.add_argument("--save_encoder_to", type=str, required=True, help="")
    _parser.add_argument(
        "--from_model",
        type=str,
        help="if supplied, training will start from that model",
        default="")

    _args = _parser.parse_args()
    main(_args)


if __name__ == "__main__":
    cli_main()
