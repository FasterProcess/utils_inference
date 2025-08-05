import torch
from typing import List
from ..bench.time_bench import test_time
from ..common.global_values import GlobalValues


class VideoData:
    def __init__(self, tensor: torch.Tensor, format="NCHW", indexs=None, scale=None):
        """
        tensor: data to manager
        format: data's format, NCHW or NHWC
        indexs: a tag to record where it come from
        scale: 1/sclae to norm to float32
        """
        self.data_ = tensor
        self.format_ = format.upper()
        self.indexs = indexs

        self.scale = scale

    def set_scale(self, scale):
        self.scale = scale

    def is_norm(self) -> bool:
        if (
            self.data_.dtype == torch.uint8
            or self.data_.dtype == torch.uint16
            or self.data_.dtype == torch.uint32
            or self.data_.dtype == torch.uint64
        ):
            return False
        else:
            return True

    def try_norm(self, dtype=torch.float32, allow_deduce=True) -> bool:
        if self.is_norm():
            self.data_ = self.data_.to(dtype)
            return True

        scale = self.scale
        if scale is None and allow_deduce:
            if self.data_.dtype == torch.uint8:
                scale = 2**8 - 1
            elif self.data_.dtype == torch.uint16:
                scale = 2**16 - 1
            elif self.data_.dtype == torch.uint32:
                scale = 2**32 - 1
            elif self.data_.dtype == torch.uint64:
                scale = 2**64 - 1
        if scale is None:
            return False
        else:
            self.scale = scale
            self.data_ = (self.data_.to(torch.float32) / scale).to(dtype)
            return True

    def try_denorm(self, img_type=None, scale=None) -> bool:
        if not self.is_norm():
            return True

        if scale is None:
            scale = self.scale

        if scale is None:
            return False

        self.data_ = self.data_.to(torch.float32) * scale
        if img_type is None:
            if scale < 2**8:
                img_type = torch.uint8
            elif scale < 2**16:
                img_type = torch.uint16
            elif scale < 2**32:
                img_type = torch.uint32
            elif scale < 2**64:
                img_type = torch.uint64
        self.data_ = self.data_.to(img_type)
        return True

    def to_device(self, device):
        self.data_ = self.data_.to(device)
        return self

    def to_format(self, format="NCHW"):
        if format.upper() == "NCHW":
            self.to_nchw()
        else:
            self.to_nhwc()
        return self

    def to_nchw(self):
        if self.format == "NCHW":
            pass
        else:
            self.data_ = self.data_.contiguous().permute(0, 3, 1, 2)
            self.format_ = "NCHW"
        return self

    def to_nhwc(self):
        if self.format == "NHWC":
            pass
        else:
            self.data_ = self.data_.contiguous().permute(0, 2, 3, 1)
            self.format_ = "NHWC"
        return self

    @test_time(enable=GlobalValues.ENABLE_PER)
    @torch.no_grad()
    def crop(self, bbox: List[int] = [0, 0, -1, -1]):
        if bbox[2] < 0:
            bbox[2] = self.width
        if bbox[3] < 0:
            bbox[3] = self.height

        if self.format == "NHWC":
            return VideoData(
                self.data_[
                    :, bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :
                ],
                format=self.format,
                scale=self.scale,
            )
        else:
            return VideoData(
                self.data_[
                    :, :, bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]
                ],
                format=self.format,
                scale=self.scale,
            )

    @test_time(enable=GlobalValues.ENABLE_PER)
    @torch.no_grad()
    def splice(self, x, y, video_data):
        if video_data.format != self.format or video_data.device != self.device:
            return False

        if self.format == "NCHW":
            self.data_[:, :, y : y + video_data.height, x : x + video_data.width] = (
                video_data.data_
            )
        else:
            self.data_[:, y : y + video_data.height, x : x + video_data.width, :] = (
                video_data.data_
            )
        return True

    @property
    def data(self) -> torch.Tensor:
        return self.data_

    @data.setter
    def data(self, value):
        self.data_ = value

    @property
    def format(self):
        return self.format_.upper()

    @property
    def device(self):
        return self.data_.device

    @property
    def dtype(self):
        return self.data_.dtype

    @property
    def shape(self):
        return self.data_.shape

    @property
    def shape_nchw(self):
        return [self.num_frame, self.channel, self.height, self.width]

    @property
    def shape_nhwc(self):
        return [self.num_frame, self.height, self.width, self.channel]

    @property
    def num_frame(self):
        return self.shape[self.format.index("N")]

    @property
    def channel(self):
        return self.shape[self.format.index("C")]

    @property
    def height(self):
        return self.shape[self.format.index("H")]

    @property
    def width(self):
        return self.shape[self.format.index("W")]
