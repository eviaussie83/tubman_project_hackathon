import cv2
import numpy as np
from scipy.stats import norm
import cvlib as cv
from cvlib.object_detection import draw_bbox


def frame_stream(file_name):
    vidcap = cv2.VideoCapture(file_name)
    success = True
    counter = 0
    while success:
        success, image = vidcap.read()
        if success:
            yield (image, vidcap.get(cv2.CAP_PROP_POS_MSEC), counter)
            counter += 1
    vidcap.release()


def iter_pairs(items):
    items_iter = iter(items)
    prev = next(items_iter)

    for item in items_iter:
        yield prev, item
        prev = item


def diff_stream(stream_of_frames):
    for ((im1, time1, counter1), (im2, time2, counter2)) in iter_pairs(stream_of_frames):
        yield {"diff": im1 - im2,
               "start_time": time1,
               "end_time": time2,
               "first_frame_index": counter1,
               "second_frame_index": counter2}


def moving_window_iterator(iterable, n):
    data = []
    iterator = iter(iterable)
    while len(data) < n:
        data.append(iterator.next())
    yield data
    for element in iterator:
        data = data[1:] + [element]
        yield data


def frame_diffs(file_name):
    return diff_stream(frame_stream(file_name))


def frame_moving_window(file_name, n):
    return moving_window_iterator(frame_diffs(file_name), n)


def moving_window_diff_norms(file_name, n):
    for window in frame_moving_window(file_name, n):
        diff_average = np.sum(frame_info["diff"].flatten() for frame_info in window) / float(n)
        yield {
            "norm": np.linalg.norm(diff_average),
            "start_time": window[0]["start_time"],
            "end_time": window[-1]["end_time"],
            "first_frame_index": window[0]["first_frame_index"],
            "final_frame_index": window[-1]["second_frame_index"]
        }


def find_unflattened_interesting_times(file_name, n, threshold):
    diff_norm_data = list(moving_window_diff_norms(file_name, n))
    diff_norms = [datum["norm"] for datum in diff_norm_data]
    mu = np.average(diff_norms)
    sigma2 = np.var(diff_norms)
    cut_off = norm.ppf(threshold, loc=mu, scale=np.sqrt(sigma2))
    return [{"start_time": datum["start_time"],
             "end_time": datum["end_time"],
             "first_frame_index": datum["first_frame_index"],
             "final_frame_index": datum["final_frame_index"]}
            for datum in diff_norm_data if datum["norm"] > cut_off]


def flatten_times(data):
    current_ff_info = None
    for frame_info in data:
        if current_ff_info is None:
            current_ff_info = frame_info
        else:
            if frame_info["first_frame_index"] <= current_ff_info["final_frame_index"]:
                current_ff_info["final_frame_index"] = frame_info["final_frame_index"]
                current_ff_info["end_time"] = frame_info["end_time"]
            else:
                yield current_ff_info
                current_ff_info = frame_info
    yield current_ff_info


def find_interesting_times(file_name, frames_in_window, percentage):
    return flatten_times(find_unflattened_interesting_times(file_name, frames_in_window, percentage))


def opencv_save_movie_with_frames(file_name, start_frame, end_frame):
    new_filename = "{start_frame}_{end_frame}_{file_name}".format(start_frame=start_frame,
                                                                  end_frame=end_frame,
                                                                  file_name=file_name)
    input_video = cv2.VideoCapture(file_name)
    for _ in range(start_frame):
        input_video.read()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    output_video = cv2.VideoWriter(new_filename, fourcc, fps, (width, height), True)
    for _ in range(end_frame - start_frame):
        success, data = input_video.read()
        bbox, label, conf = cv.detect_common_objects(data)
        draw_bbox(data, bbox, label, conf)
        output_video.write(data)
    output_video.release()
    input_video.release()
    return new_filename


def save_interesting_times(file_name, frames_in_window=15, percentage=.90):
    for period in find_interesting_times(file_name, frames_in_window, percentage):
        print(opencv_save_movie_with_frames(file_name, period["first_frame_index"], period["final_frame_index"]))


# print(list(find_interesting_times('Surveillance_footage_of_crime_scene.mp4', 15, .90)))
# print(dir(ffmpeg))

save_interesting_times('Surveillance_footage_of_crime_scene.mp4', frames_in_window=15, percentage=.90)
