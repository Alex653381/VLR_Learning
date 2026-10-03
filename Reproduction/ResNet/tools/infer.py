import argparse
from pathlib import Path

import numpy as np
import numpy.dtypes as np_dtypes
import numpy._core.multiarray as np_multiarray
import torch
from mmengine.logging.history_buffer import HistoryBuffer
from mmpretrain.apis import ImageClassificationInferencer


BASE_DIR = Path(__file__).resolve().parents[1]
CLASSES = ['cat', 'dog']

_numpy_dtype_classes = [
    dtype_cls
    for dtype_cls in vars(np_dtypes).values()
    if isinstance(dtype_cls, type)
    and issubclass(dtype_cls, np.dtype)
]

torch.serialization.add_safe_globals([
    HistoryBuffer,
    np_multiarray._reconstruct,
    np.dtype,
    np.ndarray,
    getattr,
    np_multiarray.scalar,
    *_numpy_dtype_classes,
])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single image inference for cat/dog classification'
    )
    parser.add_argument(
        'image',
        type=Path,
        help='path to the input image',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=BASE_DIR / 'configs' / 'resnet18_catdog.py',
        help='path to the MMPretrain config',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='path to the trained checkpoint',
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='device used for inference',
    )
    parser.add_argument(
        '--show-dir',
        type=Path,
        default=None,
        help='optional directory for saving the visualization',
    )
    return parser.parse_args()


def resolve_path(path):
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def main():
    args = parse_args()

    image_path = resolve_path(args.image)
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)

    if not image_path.is_file():
        raise FileNotFoundError(f'image not found: {image_path}')

    if not config_path.is_file():
        raise FileNotFoundError(f'config not found: {config_path}')

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'checkpoint not found: {checkpoint_path}'
        )

    inferencer = ImageClassificationInferencer(
        model=str(config_path),
        pretrained=str(checkpoint_path),
        device=args.device,
        classes=CLASSES,
    )

    call_kwargs = {}
    if args.show_dir is not None:
        call_kwargs['show_dir'] = str(resolve_path(args.show_dir))

    results = inferencer(str(image_path), **call_kwargs)
    result = results[0]

    scores = result['pred_scores']

    print(f'image: {image_path}')
    print(f'pred class: {result["pred_class"]}')
    print(f'pred label: {result["pred_label"]}')
    print(f'confidence: {result["pred_score"]:.4f}')

    for class_name, score in zip(CLASSES, scores):
        print(f'{class_name}: {score:.4f}')


if __name__ == '__main__':
    main()