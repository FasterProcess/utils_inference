import os
import numpy as np
import torch
from moviepy import ImageSequenceClip, VideoFileClip, concatenate_videoclips
import os
from pathlib import Path
import shutil
import numpy as np
from .video_data import VideoData
from .video_stream_writer import VideoStreamWriter
from ..common.global_values import GlobalValues


class TensorSaveVideo:
    @staticmethod
    @torch.no_grad()
    def videodata_to_video(
        data: VideoData, path: str, fps, write_to=True, append=False
    ):
        """
        This function will automatically convert the image to RGB888 format and store it. It can accept floating-point number input.
        """
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
        writer: VideoStreamWriter,
        write_to=True,
    ):
        """
        This function is not responsible for data format conversion. Please ensure that the data matches the expected format of the writer.
        """
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
        """
        This function will automatically convert the image to RGB888 format and store it. It can accept floating-point number input.
        """
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
        """
        This function will automatically convert the image to RGB888 format and store it. It can accept floating-point number input.
        """
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

        if tensor.dtype in [torch.float16, torch.float, torch.float32, torch.float64]:
            tensor = (tensor * 255).to(torch.uint8)
        elif tensor.dtype in [torch.uint16]:
            tensor = (tensor.to(torch.float32) / 65535 * 255).to(torch.uint8)

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
        writer: VideoStreamWriter,
        write_to=True,
    ):
        """
        This function is not responsible for data format conversion. Please ensure that the data matches the expected format of the writer.
        """
        return TensorSaveVideo.torch_nhwc_to_video_stream(
            tensor=tensor.movedim(1, 3),
            writer=writer,
            write_to=write_to,
        )

    @staticmethod
    def torch_nhwc_to_video_stream(
        tensor: torch.Tensor,
        writer: VideoStreamWriter,
        write_to=True,
    ):
        """
        This function is not responsible for data format conversion. Please ensure that the data matches the expected format of the writer.
        """
        if not write_to:
            return

        tensor_numpy_nhwc = tensor.cpu().numpy()
        for idx in range(tensor_numpy_nhwc.shape[0]):
            writer.Write(np.ascontiguousarray(tensor_numpy_nhwc[idx]).tobytes())
        return None

    @staticmethod
    def resize_mp4(input_path: str, resize_to: tuple, output_path: str = None) -> int:
        import moviepy

        if output_path is None:
            tmp = Path(input_path)
            output_path = os.path.join(
                tmp.parent, tmp.stem + f"_{resize_to[-2]}_{resize_to[-1]}.mp4"
            )

        tmp = Path(output_path).parent.resolve()
        os.makedirs(tmp, exist_ok=True)

        video = moviepy.VideoFileClip(input_path)
        output_video = (
            video.resized(resize_to).with_fps(video.fps).with_duration(video.duration)
        )
        output_video.write_videofile(
            output_path, codec="libx264", threads=8, fps=video.fps
        )
        return output_path, video.n_frames

    @staticmethod
    def repeate_mp4(input_path: str, repeate_to: int, output_path: str = None):
        import moviepy

        if output_path is None:
            tmp = Path(input_path)
            output_path = os.path.join(
                tmp.parent, tmp.stem + f"_{resize_to[-2]}_{resize_to[-1]}.mp4"
            )

        tmp = Path(output_path).parent.resolve()
        os.makedirs(tmp, exist_ok=True)

        video = moviepy.VideoFileClip(input_path)
        frames = []
        index = 0
        direct = 1

        assert video.n_frames > 0, "repeate_mp4 get empty video"
        for idx in range(repeate_to):
            frames.append(np.array(video.get_frame(index / video.fps)))
            if index <= 0:
                direct = 1

            elif index >= (video.n_frames - 1):
                direct = -1

            index += direct

            if video.n_frames == 1:
                index = 0

        clip = ImageSequenceClip(frames, fps=video.fps)
        clip.write_videofile(output_path, codec="libx264", bitrate="10M")
