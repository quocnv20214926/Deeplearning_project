import cv2
import sys

from app.video_source import VideoSource


if len(sys.argv) < 2:
    print("Usage: python test_video_source.py <source>")
    sys.exit()

source = sys.argv[1]

# Convert webcam index to integer.
if source.isdigit():
    source = int(source)

video = VideoSource(source)

try:
    video.start()

    print("Video source opened successfully")

    while True:

        success, frame = video.read(timeout=1)

        if success:
            cv2.imshow("Video Source Test", frame)

        elif video.is_ended():
            print("Video file has ended")
            break

        elif video.error:
            print("Error:", video.error)
            break

        # Press Q to stop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    stopped = video.stop()

    if stopped:
        print("Video reader stopped successfully")
    else:
        print("Warning: Video reader is still running")

    cv2.destroyAllWindows()
