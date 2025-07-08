import os
import decord
from typing import List
from .audio_data import AudioData
from .time_bench import test_time
from .global_values import GlobalValues
import torch


class AudioInfo:
    def __init__(self, file_path=None, target_sample_rate=None):
        self.file_path = file_path

        self.duration = 0.0  # s
        self.channel = 0

        # real after sample
        self.length_per_channel = 0
        self.sample_rate = -1

        # file origin
        self.ori_sample_rate = -1
        self.ori_length_per_channel = 0

        os.environ["DECORD_EOF_RETRY_MAX"] = "4464640"  # 436*10240
        self.init(file_path=file_path, target_sample_rate=target_sample_rate)

    def init(self, file_path=None, target_sample_rate=None) -> bool:
        self.file_path = file_path
        if file_path is None or not os.path.exists(file_path):
            self.duration = 0.0  # s
            self.channel = 0

            # real after sample
            self.length_per_channel = 0
            self.sample_rate = -1

            # file origin
            self.ori_sample_rate = -1
            self.ori_length_per_channel = 0
            return False
        else:
            reader = decord.AudioReader(self.file_path, sample_rate=-1)

            # file origin
            self.ori_length_per_channel = reader._num_samples_per_channel
            self.ori_sample_rate = reader.sample_rate
            del reader

            if target_sample_rate is not None:
                reader = decord.AudioReader(
                    self.file_path, sample_rate=target_sample_rate
                )

            self.duration = reader._duration
            self.channel = reader._num_channels

            # real after sample
            self.length_per_channel = reader._num_samples_per_channel
            self.sample_rate = reader.sample_rate
            del reader
            return True

    # @test_time(enable=GlobalValues.ENABLE_PER)
    def load_data_by_index(
        self, indexs: List[int] = [], device=decord.cpu(0), sample_rate=1.0
    ) -> AudioData:
        indexs = [index for index in indexs if (index >= 0 and index < self.length)]
        if len(indexs) < 1:
            return None

        from torch.utils import dlpack

        reader = decord.AudioReader(
            self.file_path, ctx=device, sample_rate=self.sample_rate
        )
        ori = reader.get_batch(indexs)
        return AudioData(
            dlpack.from_dlpack(ori.to_dlpack()),
            sample_rate=self.sample_rate / sample_rate,
            tag=indexs,
        )

    # @test_time(enable=GlobalValues.ENABLE_PER)
    @torch.no_grad()
    def load_data(
        self, offset=0, max_num_frame=-1, sample_rate=1, device=decord.cpu(0)
    ):
        if max_num_frame < 0:
            max_num_frame = self.length

        indexs = range(
            offset,
            min(int(offset + sample_rate * max_num_frame), self.length),
            sample_rate,
        )[:max_num_frame]

        return self.load_data_by_index(
            list(indexs), device=device, sample_rate=sample_rate
        )

    @property
    def length(self):
        return len(self)

    @property
    def ori_length(self):
        return self.ori_length_per_channel

    def __len__(self):
        return self.length_per_channel

    def __str__(self):
        return f"file_path: {self.file_path}, duration: {self.duration}s, channel: {self.channel}, length_per_channel: {self.length_per_channel}, sample_rate: {self.sample_rate}, ori_sample_rate: {self.ori_sample_rate}, ori_length_per_channel: {self.ori_length_per_channel}"

    @property
    def valid(self):
        return self.length > 0
