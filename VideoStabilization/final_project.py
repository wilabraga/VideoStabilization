import cv2
import os
import numpy as np
from stabilization import *
import matplotlib.pyplot as plt

SOURCE = 'videos/source'
OUTPUT = 'videos/output'
EXT = ['mp4', 'avi', 'mov']

crop = [.8, .8, .8, .8]
batches = [200, 200, 100, 200]


def video_to_stills(video):
    """
	Input: video (VideoCapture object)
	Output: frames (list<np.ndarray>, shape=(num_frames x h x w x c))

	Discretize video into stills
	"""

    frames = []
    ret = True

    while ret:
        ret, frame = video.read()
        if ret:
            frames.append(frame)
    video.release()

    return frames


def stills_to_video(smooth_frames, red=False, ext='mp4v'):
    """
	Input: frames (list<np.ndarray>, shape=(num_frames x h x w x c))

	Save a video from frames
	"""

    fourcc = cv2.VideoWriter_fourcc(*ext)
    if not red:
        output = cv2.VideoWriter(os.path.join(OUTPUT, 'stabilized_' + video_name),
                                fourcc, fps, (smooth_frames[0].shape[1], smooth_frames[0].shape[0]))
    else:
        output = cv2.VideoWriter(os.path.join(OUTPUT, 'red' + video_name),
                                 fourcc, fps, (smooth_frames[0].shape[1], smooth_frames[0].shape[0]))

    i = 0
    for frame in smooth_frames:
        output.write(frame)
        i += 1

    output.release()


def plot_deltas(C, P):
    """
	Input: dx (np.array, shape=(num_frames - 1)),
		   dy (np.array, shape=(num_frames - 1))

	Plot absolute dx and dy motion as a function of the frame index
	"""

    dx = C[:, 0, 2]
    dy = C[:, 1, 2]
    s_dx = P[:, 0, 2]
    s_dy = P[:, 1, 2]

    plt.figure(0)
    plt.plot(np.arange(dx.shape[0]), dx, label='original')
    plt.plot(s_dx, label='smoothed')
    plt.legend(loc='best')
    plt.xlabel('Frame Index')
    plt.ylabel('Horizontal Motion')

    plt.figure(1)
    plt.plot(np.arange(dy.shape[0]), dy, label='original')
    plt.plot(s_dy, label='smoothed')
    plt.legend(loc='best')
    plt.xlabel('Frame Index')
    plt.ylabel('Vertical Motion')
    plt.show()


if __name__ == "__main__":

    # Open directory
    videos = [os.path.relpath(x) for x in os.listdir(SOURCE) if os.path.splitext(x)[-1][1:].lower() in EXT]

    # Process each video in source folder
    for i in range(0, len(videos)):

        video_name = videos[i]
        video = cv2.VideoCapture(os.path.join(SOURCE, video_name))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print('Reading in Video ' + video_name + '...')
        frames = video_to_stills(video)

        print('Finding Motion...')
        if i < len(crop) and i < len(batches):
            C, P, B = smooth_path(frames, crop_ratio=crop[i], batch=batches[i])
        else:
            C, P, B = smooth_path(frames)

        # print('Plotting...')
        # plot_deltas(C, P)

        print('Stabilizing...')
        if i < len(crop):
            reds, smooth_frames = stabilize(frames, B, crop_ratio=crop[i])
        else:
            reds, smooth_frames = stabilize(frames, B)

        print('Processing...')
        stills_to_video(smooth_frames)
        # stills_to_video(reds, red=True)

        print('Finished Stabilizing ' + video_name + '!\n')
