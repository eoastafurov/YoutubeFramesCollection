import pafy
import cv2
import os
import json
import numpy as np
from tqdm import tqdm
import random
import warnings

random.seed(2021)


frames_skipping_table = {
    'extremely low': 900,
    'super super low': 300,
    'super low': 180,
    'very low': 120,
    'low': 80,
    'medium': 50,
    'high': 30,
    'very high': 15,
    'super super high': 10
}


class YTVideo:
    def __init__(self, video_description: {}):
        # Information from json
        self.video_url = video_description['link']
        self.begin_tc = video_description['begin']
        self.end_tc = video_description['end']
        self.label = video_description['type']
        self.frames_density = video_description['useful frames density']
        self.bad_codec = video_description['bad codec']

        # Processing information
        self.frames_between_tc = None

        # Supported information
        self.videocap = None
        self.whole_video_size = None
        self.stream = None
        self.stream_resolution = None
        self.fps = None
        self.total_frames_count = None
        self.begin_frame = None
        self.end_frame = None
        self.title = None

    def convert_tc_to_frame(self, tc: str) -> int:
        return int(np.array([int(num) for num in tc.split(':')]) @ np.array([60**2, 60, 1])) * int(self.fps)

    def get_frames_limits(self) -> (int, int):
        return self.convert_tc_to_frame(self.begin_tc), self.convert_tc_to_frame(self.end_tc)

    def save_frames(self, path_to_dataset_root: str):
        def gen_rand_name():
            return path_to_dataset_root \
                   + self.label + '/' \
                   + str(int(random.random() * 10**6)) \
                   + '.jpg'
        assert self.frames_between_tc is not None
        assert len(self.frames_between_tc) > 0
        assert os.path.exists(path_to_dataset_root)
        assert path_to_dataset_root.endswith('/')

        # frames_to_save = self.frames_between_tc[::frames_skipping_table[self.frames_density]]
        frames_to_save = self.frames_between_tc

        for frame in tqdm(frames_to_save, desc='Saving frames...', colour='blue'):
            name = gen_rand_name()
            while os.path.exists(name):
                name = gen_rand_name()
            cv2.imwrite(name, frame)


class JsonParser:
    def __init__(self, path_to_json: str):
        with open(path_to_json, 'r') as file:
            self.path_to_json = path_to_json
            self.videos = json.load(file)['videos']

    def get_next_video(self) -> (YTVideo, int):
        for i in range(len(self.videos)):
            if self.videos[i]['is_processed'] is False:
                return YTVideo(self.videos[i]), i
        return None, None

    def set_processed_status(self, idx: int):
        self.videos[idx]['is_processed'] = True
        self.dump_into_json()

    def dump_into_json(self):
        with open(self.path_to_json, 'w') as file:
            json.dump({'videos': self.videos}, file, indent=4)


class FramesProcessing:
    @staticmethod
    def process_video(ytvideo: YTVideo) -> YTVideo:
        ytvideo.stream = pafy.new(ytvideo.video_url, ydl_opts={"--no-check-certificate": True}).videostreams[0]
        ytvideo.stream_resolution = ytvideo.stream.resolution
        ytvideo.whole_video_size = ytvideo.stream.get_filesize() / 1024 ** 2
        ytvideo.videocap = cv2.VideoCapture(ytvideo.stream.url)
        ytvideo.fps = ytvideo.videocap.get(cv2.CAP_PROP_FPS)
        ytvideo.total_frames_count = ytvideo.videocap.get(cv2.CAP_PROP_FRAME_COUNT)
        ytvideo.begin_frame, ytvideo.end_frame = ytvideo.get_frames_limits()
        ytvideo.videocap.set(cv2.CAP_PROP_POS_FRAMES, ytvideo.begin_frame)
        ytvideo.title = pafy.new(ytvideo.video_url).title
        ytvideo.frames_between_tc = []

        print('Working with video: {}'.format(ytvideo.title))

        assert ytvideo.stream_resolution == '256x144'
        assert ytvideo.fps != 0

        frames_step = frames_skipping_table[ytvideo.frames_density]
        frames_to_read = (ytvideo.end_frame - ytvideo.begin_frame) // frames_step
        curr_num_of_read = 0

        # frames_to_read = (ytvideo.end_frame - ytvideo.begin_frame)
        # curr_num_of_read = 0

        progress_bar = tqdm(total=frames_to_read, desc='Frames gathering...', colour='green')
        while ytvideo.videocap.isOpened():
            status, frame = ytvideo.videocap.read()
            progress_bar.update(1)
            curr_num_of_read += 1
            next_frame = ytvideo.begin_frame + frames_step * curr_num_of_read

            ytvideo.videocap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
            if status is True and curr_num_of_read < frames_to_read:
                ytvideo.frames_between_tc.append(frame)
            elif status is False:
                warnings.warn("Can't read video till the end")
                break
            else:
                break
        ytvideo.videocap.release()
        progress_bar.close()

        return ytvideo


def process():
    jp = JsonParser('links_n_timecodes.json')

    while True:
        ytvideo, idx = jp.get_next_video()

        if ytvideo is None:
            break

        try:
            ytvideo = FramesProcessing.process_video(ytvideo)
            ytvideo.save_frames('Dataset/')
        except AssertionError:
            jp.videos[idx]['bad codec'] = True

        jp.set_processed_status(idx)

    print('Done!')


if __name__ == '__main__':
    process()



