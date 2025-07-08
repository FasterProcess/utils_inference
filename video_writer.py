from .global_values import GlobalValues
from .time_bench import test_time
import torch
from moviepy import ImageSequenceClip, VideoFileClip, concatenate_videoclips
import os
from pathlib import Path
import shutil
import ffmpeg
import numpy as np
import subprocess
from .video_data import VideoData


class FFmpegWriter:
    def __init__(
        self,
        path,
        width,
        height,
        fps,
        input_fmt="rgb24",
        vcodec="libx264",
        output_fmt="yuv420p",
        bitrate="10M",
        loglevel="warning",
        writer=None,
        audio_file=None,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.need_add_audio = False
        self.audio_path = audio_file
        self.save_path = path
        if audio_file is not None and os.path.exists(audio_file):
            self.need_add_audio = True
            suffix = Path(self.save_path).suffix
            self.tmp_save_path = self.save_path.replace(suffix, "_tmp~" + suffix)
        else:
            self.need_add_audio = False
            self.tmp_save_path = self.save_path

        if writer is None:
            streams = []
            video_stream = ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt=input_fmt,
                s="{}x{}".format(width, height),
                r=fps,
            )
            streams.append(video_stream)

            # insert audio
            extern_args = {}
            # if audio_file is not None and os.path.exists(audio_file):
            #     audio_info = ffmpeg.probe(audio_file)
            #     audio_duration = float(audio_info["format"]["duration"])
            #     audio_stream = ffmpeg.input(audio_file).audio

            #     if audio_duration > duration:
            #         audio_stream = audio_stream.filter("atrim", end=duration)
            #     else:
            #         audio_stream = audio_stream.filter("apad", whole_dur=duration)

            #     extern_args["acodec"] = "aac"
            #     # extern_args["shortest"] = None
            #     streams.append(audio_stream)

            self.process = (
                ffmpeg.output(
                    *streams,
                    self.tmp_save_path,
                    vcodec=vcodec,
                    pix_fmt=output_fmt,
                    loglevel=loglevel,
                    video_bitrate=bitrate,
                    r=fps,
                    **extern_args,
                )
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        else:
            self.process = writer
        self.is_close = False

    def set_process(self, writer):
        self.process = writer

    def Write(self, data: bytes) -> bool:
        if self.is_close:
            return False

        try:
            self.process.stdin.write(data)
        except Exception as ex:
            return False
        return True

    @test_time(enable=GlobalValues.ENABLE_PER)
    @staticmethod
    def add_audio_to_video(video_path, audio_path, output_path):
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-filter_complex",
            "[1:a]apad",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-loglevel",
            "quiet",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    def Close(self):
        if not self.is_close:
            self.process.stdin.close()
            self.process.wait()
            if self.need_add_audio:
                if os.path.exists(self.save_path):
                    os.remove(self.save_path)
                FFmpegWriter.add_audio_to_video(
                    self.tmp_save_path, self.audio_path, self.save_path
                )
                os.remove(self.tmp_save_path)

            self.is_close = True

    def __del__(self):
        self.Close()


class TensorSaveVideo:
    @staticmethod
    @torch.no_grad()
    def videodata_to_video(
        data: VideoData, path: str, fps, write_to=True, append=False
    ):
        if data.format == "NCHW":
            TensorSaveVideo.torch_nchw_to_video(
                tensor=data.data, path=path, fps=fps, write_to=write_to, append=append
            )
        else:
            TensorSaveVideo.torch_nhwc_to_video(
                tensor=data.data, path=path, fps=fps, write_to=write_to, append=append
            )

    @staticmethod
    @torch.no_grad()
    def videodata_to_stream(
        data: VideoData,
        writer: FFmpegWriter,
        write_to=True,
    ):
        if data.format == "NCHW":
            return TensorSaveVideo.torch_nchw_to_video_stream(
                tensor=data.data,
                writer=writer,
                write_to=write_to,
            )
        else:
            return TensorSaveVideo.torch_nhwc_to_video_stream(
                tensor=data.data,
                writer=writer,
                write_to=write_to,
            )

    @staticmethod
    @torch.no_grad()
    def torch_nchw_to_video(
        tensor: torch.Tensor, path: str, fps, write_to=True, append=False
    ):
        return TensorSaveVideo.torch_nhwc_to_video(
            tensor=tensor.movedim(1, 3),
            path=path,
            fps=fps,
            write_to=write_to,
            append=append,
        )

    @staticmethod
    @torch.no_grad()
    def torch_nhwc_to_video(
        tensor: torch.Tensor, path: str, fps, write_to=True, append=False
    ):
        tmp = Path(path).parent.resolve()
        os.makedirs(tmp, exist_ok=True)

        clips = []
        raw_clip = None
        if os.path.exists(path) and append:
            raw_clip = VideoFileClip(path)
            clips.append(raw_clip)
            fps = clips[0].fps
            if GlobalValues.DEBUG:
                print(f"append data to raw video:{clips[0].reader.nframes}")

        if tensor.shape[3] == 1:
            tensor = tensor.repeat(1, 1, 1, 3)  # repeat gray image to RGB image

        if tensor.dtype != torch.uint8:
            tensor = (tensor * 255).to(torch.uint8)
        tensor_numpy_nhwc = tensor.cpu().numpy()
        new_clip = ImageSequenceClip(
            list(np.ascontiguousarray(tensor_numpy_nhwc)), fps=fps
        )
        clips.append(new_clip)

        clip = concatenate_videoclips(clips)
        if write_to:
            clip.write_videofile(
                "_cache_.mp4" if raw_clip is not None else path, codec="libx264"
            )

            if raw_clip is not None:
                raw_clip.close()
                clip.close()
                shutil.move("_cache_.mp4", path)
        return None

    @staticmethod
    @torch.no_grad()
    def torch_nchw_to_video_stream(
        tensor: torch.Tensor,
        writer: FFmpegWriter,
        write_to=True,
    ):
        return TensorSaveVideo.torch_nhwc_to_video_stream(
            tensor=tensor.movedim(1, 3),
            writer=writer,
            write_to=write_to,
        )

    @staticmethod
    def torch_nhwc_to_video_stream(
        tensor: torch.Tensor,
        writer: FFmpegWriter,
        write_to=True,
    ):
        if not write_to:
            return

        tensor_numpy_nhwc = tensor.cpu().numpy().astype(np.uint8)
        for idx in range(tensor_numpy_nhwc.shape[0]):
            writer.Write(np.ascontiguousarray(tensor_numpy_nhwc[idx]).tobytes())
        return None
