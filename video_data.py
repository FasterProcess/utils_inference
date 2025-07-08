import torch
from typing import List
from .time_bench import test_time
from .global_values import GlobalValues


class VideoData:
    def __init__(self, tensor: torch.Tensor, format="NCHW", tag=None):
        """
        tensor: data to manager
        format: data's format, NCHW or NHWC
        indexs: a tag to record where it come from
        """
        self.data_ = tensor
        self.format_ = format.upper()
        self.tag = tag

    @torch.no_grad()
    def to_device(self, device, inplace=True):
        if inplace:
            self.data_ = self.data_.to(device)
            return self
        else:
            return VideoData(
                tensor=self.data_.to(device),
                format=self.format_,
                tag=self.tag,
            )

    @torch.no_grad()
    def to_format(self, format="NCHW", inplace=True):
        if format.upper() == "NCHW":
            return self.to_nchw(inplace=inplace)
        else:
            return self.to_nhwc(inplace=inplace)

    @torch.no_grad()
    def to_nchw(self, inplace=True):
        if self.format == "NCHW":
            if inplace:
                return self
            else:
                return VideoData(
                    tensor=self.data_[:],
                    format=self.format_,
                    tag=self.tag,
                )
        else:
            if inplace:
                self.data_ = self.data_.movedim(3, 1)
                # contiguous().permute(0, 3, 1, 2)
                self.format_ = "NCHW"
                return self
            else:
                return VideoData(
                    tensor=self.data_.movedim(3, 1),
                    format="NCHW",
                    tag=self.tag,
                )

    @torch.no_grad()
    def to_nhwc(self, inplace=True):
        if self.format == "NHWC":
            if inplace:
                return self
            else:
                return VideoData(
                    tensor=self.data_[:],
                    format=self.format_,
                    tag=self.tag,
                )
        else:
            if inplace:
                self.data_ = self.data_.movedim(1, 3)
                # contiguous().permute(0, 3, 1, 2)
                self.format_ = "NHWC"
                return self
            else:
                return VideoData(
                    tensor=self.data_.movedim(1, 3),
                    format="NHWC",
                    tag=self.tag,
                )

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
            )
        else:
            return VideoData(
                self.data_[
                    :, :, bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]
                ],
                format=self.format,
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
