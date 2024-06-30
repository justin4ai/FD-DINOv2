import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import inspect

import sys
sys.path.append("./")

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from dino import DINOv2Encoder
from utils.load_encoder import load_encoder
from utils.dataset import ImagePathDataset

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


from src.pytorch_fd.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=200, help="Batch size to use")
parser.add_argument("--num-workers", type=int)
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use. Like cuda, cuda:0 or cpu")
parser.add_argument("--dims", type=int, default=1024,help=("Dimensionality of DINOv2 features to use."))
parser.add_argument(
    "--save-stats",
    action="store_true",
    help=(
        "Generate an npz archive from a directory of "
        "samples. The first path is used as input and the "
        "second as output."
    ),
)
parser.add_argument(
    "path",
    type=str,
    nargs=2,
    help=("Paths to the generated images or " "to .npz statistic files"),
)

parser.add_argument('--clean_resize', action='store_true',
                    help='Use clean resizing (from pillow)')

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}





def get_activations(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    model.eval()

    if batch_size > len(files):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.Compose([TF.Resize((224, 224)), TF.ToTensor()]))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        pred = pred.cpu().numpy()
        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fd calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):

    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1):
    if path.endswith(".npz"):
        with np.load(path) as f:
            m, s = f["mu"][:], f["sigma"][:]
    else:
        path = pathlib.Path(path)
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )


        m, s = calculate_activation_statistics(
            files, model, batch_size, dims, device, num_workers
        )

    return m, s


def calculate_fd_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FD-DINOv2 of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    model = load_encoder("dinov2", device, ckpt = None, arch = None,
                        clean_resize = False,
                        sinception = False,
                        depth = 0)

    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers
    )
    m2, s2 = compute_statistics_of_path(
        paths[1], model, batch_size, dims, device, num_workers
    )
    fd_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fd_value


def save_fd_stats(paths, batch_size, device, dims, num_workers=1):
    """Saves FD-DINOv2 statistics of one path"""
    if not os.path.exists(paths[0]):
        raise RuntimeError("Invalid path: %s" % paths[0])

    if os.path.exists(paths[1]):
        raise RuntimeError("Existing output file: %s" % paths[1])

    model = load_encoder("dinov2", device, ckpt = None, arch = None,
                        clean_resize = False,
                        sinception = False,
                        depth = 0)

    print(f"Saving statistics for {paths[0]}")

    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers
    )

    np.savez_compressed(paths[1], mu=m1, sigma=s1)


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:

            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    if args.save_stats:
        save_fd_stats(args.path, args.batch_size, device, args.dims, num_workers)
        return

    fd_value = calculate_fd_given_paths(
        args.path, args.batch_size, device, args.dims, num_workers
    )
    print("FD-DINOv2: ", fd_value)


if __name__ == "__main__":

    main()