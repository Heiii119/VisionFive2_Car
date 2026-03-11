#!/usr/bin/env python3
import argparse
import sys

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

def build_pipeline(args) -> str:
    # Notes:
    # - v4l2src reads USB camera frames
    # - videoconvert/videoscale ensure raw video format
    # - hailonet runs inference using your .hef
    # - hailofilter loads a post-process .so (depends on your model/demo)
    # - hailooverlay draws boxes/labels on the video
    #
    # You MUST set paths that match your installed Tappas postprocess and your model HEF.
    return (
        f"v4l2src device={args.device} ! "
        f"video/x-raw,framerate={args.fps}/1 ! "
        f"videoconvert ! videoscale ! "
        f"video/x-raw,width={args.width},height={args.height} ! "
        f"queue ! "
        f"hailonet hef-path={args.hef} batch-size=1 ! "
        f"queue ! "
        f"hailofilter so-path={args.postprocess_so} function-name={args.function_name} ! "
        f"queue ! "
        f"hailooverlay ! "
        f"videoconvert ! "
        f"autovideosink sync=false"
    )

def on_bus_message(bus, message, loop):
    mtype = message.type
    if mtype == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        print(f"[GStreamer ERROR] {err}\n[debug] {dbg}", file=sys.stderr)
        loop.quit()
    elif mtype == Gst.MessageType.EOS:
        loop.quit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--hef", required=True, help="Path to your model .hef")
    parser.add_argument("--postprocess-so", required=True, help="Path to Tappas post-process .so")
    parser.add_argument("--function-name", default="postprocess", help="Function name inside the .so")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    Gst.init(None)
    pipeline_str = build_pipeline(args)
    print("Pipeline:\n", pipeline_str)

    pipeline = Gst.parse_launch(pipeline_str)
    bus = pipeline.get_bus()

    loop = GLib.MainLoop()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
