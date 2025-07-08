import os
import decord
from typing import List
from .video_data import VideoData
import ffmpeg
from .time_bench import test_time
from .global_values import GlobalValues
import torch


class VideoInfo:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.num_frame = 0
        self.channel = 0
        self.width = 0
        self.height = 0
        self.fps = 0
        self.bit_rate = 0
        self.duration = 0

        os.environ["DECORD_EOF_RETRY_MAX"] = "4464640"  # 436*10240
        self.init(file_path=file_path)

    def init(self, file_path=None) -> bool:
        self.file_path = file_path
        if file_path is None or not os.path.exists(file_path):
            self.raw_num_frame = 0
            self.num_frame = 0
            self.channel = 0
            self.width = 0
            self.height = 0
            self.fps = 0
            self.bit_rate = 0
            self.duration = 0
        else:
            reader = decord.VideoReader(self.file_path, ctx=decord.cpu(0))
            self.fps = reader.get_avg_fps()
            self.num_frame = len(reader)

            self.height, self.width, self.channel = reader[0].shape
            del reader

            info = ffmpeg.probe(self.file_path)
            self.bit_rate = int(info["streams"][0]["bit_rate"]) // 1000000
            self.duration = float(info["format"]["duration"])
            del info

    @test_time(enable=GlobalValues.ENABLE_PER)
    def load_data_by_index(
        self, indexs: List[int] = [], device=decord.cpu(0), format="NHWC"
    ) -> VideoData:
        indexs = [index for index in indexs if (index >= 0 and index < self.num_frame)]
        if len(indexs) < 1:
            return None

        from torch.utils import dlpack

        reader = decord.VideoReader(self.file_path, ctx=device)
        ori = reader.get_batch(indexs)
        return VideoData(
            dlpack.from_dlpack(ori.to_dlpack()), format="NHWC", tag=indexs
        ).to_format(format)

    @test_time(enable=GlobalValues.ENABLE_PER)
    @torch.no_grad()
    def load_data(self, offset, max_num_frame, sample_rate=1, device=decord.cpu(0)):
        if max_num_frame < 0:
            max_num_frame = self.num_frame - offset

        indexs = range(
            offset,
            min(offset + sample_rate * max_num_frame, self.num_frame),
            sample_rate,
        )[:max_num_frame]

        return self.load_data_by_index(list(indexs), device=device)

    @property
    def shape_nhwc(self):
        return [self.num_frame, self.height, self.width, self.channel]

    @property
    def shape_nchw(self):
        return [self.num_frame, self.channel, self.height, self.width]

    @property
    def valid(self):
        return self.num_frame > 0
