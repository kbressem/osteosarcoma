import logging
from typing import Dict, Hashable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F  # noqa
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms import Lambdad, MapTransform, Transform
from monai.transforms.utils import TransformBackends, ensure_tuple_rep
from monai.utils.type_conversion import convert_data_type

logger = logging.getLogger(__name__)


class AdaptAffineMatrix(Transform):
    """
    Change negative spacing values to positive
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self) -> None:
        pass

    def __call__(
        self, image: NdarrayOrTensor, meta_dict: Optional[Dict] = None
    ) -> Tuple[NdarrayOrTensor, Dict]:
        image.meta["affine"][0:3, 0:3] = abs(image.meta["affine"][0:3, 0:3])
        image.meta["original_affine"][0:3, 0:3] = abs(image.meta["original_affine"][0:3, 0:3])
        if meta_dict is not None:
            meta_dict["affine"][0:3, 0:3] = abs(meta_dict["affine"][0:3, 0:3])
            meta_dict["original_affine"][0:3, 0:3] = abs(meta_dict["original_affine"][0:3, 0:3])
        return image, meta_dict


class AdaptAffineMatrixd(MapTransform):
    """Dictionary-based wrapper of :py:class:`AdaptAffineMatrix`"""

    backend = AdaptAffineMatrix.backend

    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
    ) -> None:

        super().__init__(keys)
        self.adjuster = AdaptAffineMatrix()
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, meta_key, meta_key_postfix in zip(
            self.keys, self.meta_keys, self.meta_key_postfix
        ):
            values, meta = self.adjuster(
                d[key],
                d[meta_key or f"{key}_{meta_key_postfix}"],
            )
            d[key] = values
            d[meta_key or f"{key}_{meta_key_postfix}"] = meta
        return d


class EmptyArrayError(ValueError):
    pass


class RaiseOnEmptyd(MapTransform):
    """Debug-helper. Raises if an array is all zeros."""

    backend = AdaptAffineMatrix.backend

    def __init__(
        self,
        keys: KeysCollection,
    ) -> None:

        super().__init__(keys)

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            if d[key].sum() < 1:
                raise EmptyArrayError(d[key].meta["filename_or_obj"])
        return d


class SqueezeAffined(Lambdad):
    def __init__(self, keys, allow_missing_keys: True):
        def squeeze_affine(x):
            x.meta["affine"] = x.meta["affine"].squeeze()
            return x

        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys, func=squeeze_affine)


class MatchSize(Transform):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

    def __call__(self, image: NdarrayOrTensor, reference: NdarrayOrTensor) -> NdarrayOrTensor:
        image_, prev_type, device = convert_data_type(image, torch.Tensor)
        reference_size = reference.shape[1:]
        image_ = image_.unsqueeze(0)
        image_ = F.interpolate(image_, reference_size, mode=self.mode)
        image_ = image_[0]  # remove batch dim
        image, *_ = convert_data_type(image_, prev_type, device)
        return image


class MatchSized(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mode: Union[Tuple, List[str]],
        reference_key: str,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.reference_key = reference_key

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, mode in zip(self.keys, self.mode):
            self.adjuster = MatchSize(mode=mode)
            d[key] = self.adjuster(d[key], d[self.reference_key])
        return d
