from .video_info_ffmpeg import VideoInfoFfmpeg
import os
import ffmpeg
import json
import os
import ffmpeg
import subprocess
from .video_stream_writer import VideoStreamWriter


class VideoStreamWriterFfmpeg(VideoStreamWriter):
    def __init__(
        self, save_path, writer=None, refer_file=None, loglevel="warning", **kwargs
    ):
        self.empty_stream = True

        self.path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.is_close = False

        if writer is not None:
            self.process = writer
            return

        args = {}
        for k, v in kwargs.items():
            args[k] = v

        profile = self.construct_profile(args, refer_file=refer_file)

        for k, v in self.get_default_arg().items():
            if k not in profile:
                profile[k] = v

        assert (
            "width" in profile and "height" in profile
        ), "need width * height while set writer of file"

        # print(f"profile: {profile}")

        cmd_scale = {}
        if "cmd_scale" in profile:
            cmd_scale = profile["cmd_scale"]

        addtion_cmd = {}
        if "vcodec" in profile:
            if profile["vcodec"] == "prores_ks":
                assert (
                    "ffmpeg_profile_value" in profile
                ), "need ffmpeg_profile_value if vcodec==prores_ks"
                addtion_cmd["profile:v"] = profile["ffmpeg_profile_value"]

        self.process = (
            ffmpeg.input(
                "pipe:",
                format=profile.get("format", "rawvideo"),
                pix_fmt=profile.get("rgb_fmt", "rgb24"),
                s="{}x{}".format(int(profile["width"]), int(profile["height"])),
                r=profile.get("fps", 30),
            )
            .filter("scale", **cmd_scale)
            .output(
                save_path,
                vcodec=profile.get("vcodec", "libx264"),
                pix_fmt=profile.get("pix_fmt", "rgb24"),
                loglevel=loglevel,
                video_bitrate=max(profile.get("bit_rate", 10000000), 10000000),
                r=profile.get("fps", 30),
                **addtion_cmd,
                **profile["cmd"],
                **profile["field_order_kwargs"],
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        # print("FFmpeg command:", " ".join(self.process.compile()))

        self.is_close = False

    def get_default_arg(self) -> dict:
        return {
            "format": "rawvideo",
            "pix_fmt": "yuv420p",
            "rgb_fmt": "rgb24",
            "fps": 30,
            "vcodec": "libx264",
            "bit_rate": 10000000,
            "field_order": "progressive",
            "field_order_kwargs": {},
        }

    def construct_profile(self, args: dict, refer_file=None) -> dict:
        profile = {}
        if refer_file is not None:
            profile = VideoInfoFfmpeg(refer_file).meta_data

        for k, v in args.items():
            profile[k] = v

        if "cmd_scale" not in profile:
            if profile.get("color_space", None) is not None:
                if "bt601" in profile["color_space"]:
                    color_matrix = "bt601"
                elif "bt470" in profile["color_space"]:
                    color_matrix = "bt470"
                else:
                    color_matrix = profile["color_space"]
                cmd_scale = {
                    "in_color_matrix": "{}".format(color_matrix),
                    "out_color_matrix": "{}".format(color_matrix),
                }
                profile["cmd_scale"] = cmd_scale

        if "cmd" not in profile:
            cmd = {}
            if profile.get("color_space", None) is not None:
                cmd["colorspace"] = profile["color_space"]
            if profile.get("color_transfer", None) is not None:
                cmd["color_trc"] = profile["color_transfer"]
            if profile.get("color_primaries", None) is not None:
                cmd["color_primaries"] = profile["color_primaries"]
            profile["cmd"] = cmd

        if refer_file is not None:
            probe = ffmpeg.probe(refer_file)
            video_streams = [
                stream for stream in probe["streams"] if stream["codec_type"] == "video"
            ]

            if "vcodec" not in profile:
                vcodec = video_streams[0]["codec_name"]
                if vcodec.lower() == "prores":
                    vcodec = "prores_ks"
                profile["vcodec"] = vcodec

            ffmpeg_profile_value_map = {
                "unknown": "3",
                "proxy": "0",
                "lt": "1",
                "standard": "2",
                "hq": "3",
                "4444": "4",
                "4444 xq": "5",
            }

            if "ffmpeg_profile_value" not in profile:
                profile["ffmpeg_profile_value"] = ffmpeg_profile_value_map.get(
                    video_streams[0].get("profile", "unknown").lower(), "3"
                )

            """
            Determine whether the video is progressive or interlaced scanning.
            return 'progressive', 'top_field_first', 'bottom_field_first' or 'unknown_profile["field_order"]'。
            If there is no video stream in the video or ffprobe fails, return 'error'.
            """
            try:
                if "field_order" not in profile:
                    field_order = video_streams[0].get("field_order")

                    if field_order in ["top", "top_field_first", "tb", "tt", "tff"]:
                        profile["field_order"] = "top_field_first"
                    elif field_order in [
                        "bottom",
                        "bottom_field_first",
                        "bb",
                        "bt",
                        "bff",
                    ]:
                        profile["field_order"] = "bottom_field_first"
                    elif field_order == "progressive":
                        profile["field_order"] = "progressive"
                    elif field_order == "interlaced":
                        profile["field_order"] = "interlaced"
                    elif field_order is None:
                        profile["field_order"] = "progressive"
                    # else:
                    #     return "unknown_profile["field_order"]"

            except ffmpeg.Error as e:
                print(f"FFprobe Error for {refer_file}: {e.stderr.decode('utf8')}")
                raise e
            except json.JSONDecodeError:
                print(f"Failed to decode JSON from ffprobe output for {refer_file}")
                raise e
            except Exception as e:
                print(f"An unexpected error occurred during scan type detection: {e}")
                raise e

            if "field_order_kwargs" not in profile and "field_order" in profile:
                profile["field_order_kwargs"] = {}
                if profile["field_order"] in [
                    "top_field_first",
                    "bottom_field_first",
                    "interlaced",
                ]:
                    profile["field_order_kwargs"]["flags"] = "+ildct+ilme"
                    if profile["field_order"] == "top_field_first":
                        profile["field_order_kwargs"]["top"] = 1
                    elif profile["field_order"] == "bottom_field_first":
                        profile["field_order_kwargs"]["top"] = 0
                elif profile["field_order"] == "progressive":
                    pass
                else:
                    print(
                        f"Warning: Unrecognized scan type {profile['field_order']}. Defaulting to progressive processing."
                    )

        return profile

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

    def set_process(self, writer):
        self.process = writer

    def Write(self, data: bytes) -> bool:
        if self.is_close:
            return False

        try:
            self.process.stdin.write(data)
            self.empty_stream = False
        except Exception as ex:
            return False
        return True

    def Close(self):
        if not self.is_close:
            self.process.stdin.close()

            if not self.empty_stream:
                self.process.wait()
            self.is_close = True

    def __del__(self):
        self.Close()
