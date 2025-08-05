import os
from typing import List

import numpy as np
from .video_data import VideoData
import ffmpeg
from ..bench.time_bench import test_time
from ..common.global_values import GlobalValues
from .video_info import VideoInfo
from torch.utils import dlpack


class VideoInfoFfmpeg(VideoInfo):
    """
    read video by ffmpeg
    """

    def __init__(self, file_path: str = None):
        super().__init__(file_path)
        self.init(file_path=file_path)

    def init(self, file_path: str = None) -> bool:
        if file_path is None:
            self.file_path = None
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
            self.file_path = os.path.abspath(file_path)
            ret = self.get_video_meta_info(self.file_path)
            self.num_frame = ret["nb_frames"]
            self.channel = 1 if ret["pix_fmt"] in ["gray"] else 3
            self.width = ret["width"]
            self.height = ret["height"]
            self.fps = ret["fps"]
            self.bit_rate = ret["bit_rate"]

            self.pix_fmt = ret["pix_fmt"]
            self.color_space = ret["color_space"]
            self.color_transfer = ret["color_transfer"]
            self.color_primaries = ret["color_primaries"]
            self.duration = ret["duration"]

            self.addition_tag = ret
            return True

    def get_video_meta_info(self, video_path) -> dict:
        """get the meta info of the video by using ffprobe with python interface"""
        ret = {}
        probe = ffmpeg.probe(video_path)
        video_streams = [
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        ]
        has_audio = any(stream["codec_type"] == "audio" for stream in probe["streams"])
        ret["width"] = video_streams[0]["width"]
        ret["height"] = video_streams[0]["height"]
        ret["fps"] = eval(video_streams[0]["avg_frame_rate"])
        ret["audio"] = ffmpeg.input(video_path).audio if has_audio else None
        ret["pix_fmt"] = video_streams[0]["pix_fmt"]
        ret["bit_rate"] = int(video_streams[0]["bit_rate"])
        if "color_space" in video_streams[0]:
            ret["color_space"] = video_streams[0]["color_space"]
        else:
            ret["color_space"] = None

        if "color_transfer" in video_streams[0]:
            ret["color_transfer"] = video_streams[0]["color_transfer"]
        else:
            ret["color_transfer"] = None

        if "color_primaries" in video_streams[0]:
            ret["color_primaries"] = video_streams[0]["color_primaries"]
        else:
            ret["color_primaries"] = None

        # ret['audio'] = None
        try:
            ret["nb_frames"] = int(video_streams[0]["nb_frames"])
            ret["duration"] = float(video_streams[0]["duration"])
        except KeyError:  # bilibili transcoder dont have nb_frames
            ret["duration"] = float(probe["format"]["duration"])
            ret["nb_frames"] = int(ret["duration"] * ret["fps"])
        return ret

    @test_time(enable=GlobalValues.ENABLE_PER)
    def load_data_by_index(
        self, indexs: List[int] = [], device="cpu", format="NHWC"
    ) -> VideoData:
        device_str, _ = self.read_device(device)
        input_args = {}
        output_args = {}
        if device_str == "cpu":
            input_args = {}
            output_args = {}
        elif device_str == "cuda":
            input_args = {"hwaccel": "cuda", "hwaccel_output_format": "cuda"}
            output_args = {"vcodec": "h264_nvenc"}
        else:
            raise Exception(
                f"VideoInfoDecord device only support cpu, {device_str} is given"
            )

        indexs = [index for index in indexs if (index >= 0 and index < self.num_frame)]
        if len(indexs) < 1:
            return None

        input_args["ss"] = min(indexs) / self.fps
        process = (
            ffmpeg.input(self.file_path, **input_args)
            # .filter("select", f"between(n,{min(indexs)},{max(indexs)})")
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt=self.rgb_fmt,
                s=f"{self.width}x{self.height}",
                vframes=max(indexs) - min(indexs) + 1,
                loglevel="warning",
                **output_args,
            ).run_async(pipe_stdout=True, pipe_stderr=True)
        )

        buffer_size = (
            self.width * self.height * self.channel * (2 if self.is_high_bit else 1)
        )
        raw_frames = []

        while True:
            in_bytes = process.stdout.read(buffer_size)
            if not in_bytes:
                break
            raw_frames.append(in_bytes)

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            raise Exception(
                f"fail to run ffmepeg, info: {process.stderr.read().decode()}"
            )

        data_type = np.uint16 if self.is_high_bit else np.uint8
        if data_type == np.uint16:
            video_data = np.frombuffer(b"".join(raw_frames), "<u2")
        else:
            video_data = np.frombuffer(b"".join(raw_frames), dtype=np.uint8)
        num_read_frames = len(indexs)

        try:
            ori = video_data.reshape(
                (num_read_frames, self.height, self.width, self.channel)
            )[
                np.array(indexs, np.int32) - min(indexs)
            ]  # copy to set readonly-data to editable-data
        except Exception as ex:
            raise Exception(
                f"fail to read frames, bytes length error, expected frame={len(indexs)}"
            )

        return VideoData(
            dlpack.from_dlpack(ori),
            format="NHWC",
            indexs=indexs,
            scale=(2**16 - 1) if self.is_high_bit else (2**8 - 1),
        ).to_format(format)
