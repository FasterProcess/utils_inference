from pycocotools import mask as mask_util
from typing import Optional, List
from .video_data import VideoData
import json
import numpy as np
import torch
from .video_info import VideoInfo
import os
from ..bench.time_bench import test_time
from ..common.global_values import GlobalValues


class VideoInfoCoco(VideoInfo):
    """
    read coco json to mask video
    """

    def __init__(self, file_path=None):
        super().__init__(file_path)
        self.init(file_path=file_path)

    def init(self, file_path=None) -> bool:
        self.file_path = file_path
        try:
            with open(file_path, "r") as fp:
                mask_json = json.load(fp)
        except Exception as ex:
            mask_json = None
            file_path = None

        if file_path is None:
            self.raw_num_frame = 0
            self.num_frame = 0
            self.channel = 0
            self.width = 0
            self.height = 0
            self.fps = 0
            self.bit_rate = 0

            self.pix_fmt = None
            self.color_space = None
            self.color_transfer = None
            self.color_primaries = None
            self.duration = None

            self.addition_tag = None
            return False
        else:
            with open(file_path, "r") as fp:
                mask_json = json.load(fp)
            self.mask_json = mask_json
            self.num_frame = len(mask_json)

            self.height, self.width = mask_json[0]["size"]
            self.channel = 3
            self.fps = 30
            self.bit_rate = 10000000

            self.pix_fmt = "yuv420p"
            self.color_space = None
            self.color_transfer = None
            self.color_primaries = None
            self.duration = self.num_frame / self.fps

            self.addition_tag = None
        return True

    @test_time(enable=GlobalValues.ENABLE_PER)
    def load_data_by_index(
        self, indexs: List[int] = [], device="cpu", format="NHWC"
    ) -> VideoData:
        device_str, _ = self.read_device(device)
        if device_str == "cpu":
            pass
        else:
            raise Exception(
                f"VideoInfoDecord device only support cpu, {device_str} is given"
            )

        indexs = [index for index in indexs if (index >= 0 and index < self.num_frame)]
        if len(indexs) < 1:
            return None

        ori = []
        for i in indexs:
            binary_mask = mask_util.decode(
                {
                    "counts": self.mask_json[i]["counts"][0]["mask"],
                    "size": self.mask_json[i]["size"],
                }
            )
            ori.append(binary_mask)

        if format.upper() == "NHWC":
            ori = np.repeat(np.expand_dims(np.array(ori), 3), 3, axis=3)
        else:
            ori = np.repeat(np.expand_dims(np.array(ori), 1), 3, axis=1)

        return VideoData(torch.from_numpy(ori), format=format, scale=255)
