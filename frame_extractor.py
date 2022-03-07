import cv2
import argparse
from pathlib import Path
from icecream import ic


def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video')
    parser.add_argument('input', type=lambda p: Path(p).absolute(), help='input video file')
    parser.add_argument('output', type=lambda p: Path(p).absolute(), help='output directory')

    args = parser.parse_args()

    print(args.input)
    print(args.output)

    cap = cv2.VideoCapture(str(args.input))
    counter = 0
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        frame_counter += 1
        ic(frame_counter)
        if ret is False:
            break

        cv2.imshow("frame", frame)
        
        k = cv2.waitKey(30)
        if k == ord('q'):
            break
        if k == ord('s'):
            filename = "%s/frame%03d.jpg" % (args.output, counter)
            counter += 1
            cv2.imwrite(filename, frame)
            print(f"Saved frame { frame_counter } to the file { filename }")


main()
