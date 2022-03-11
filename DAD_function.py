import tensorflow as tf
import numpy as np
import cv2
import sys
import time


def load_model(path):
    return tf.keras.models.load_model(path)


def sort_buffer(temp_buffer, alert_result, map, longest_frame=0, alert_conf = 0):
    for item_class, (item_frame, item_conf) in temp_buffer.items():
        if item_frame > longest_frame:
            longest_frame = item_frame
            alert_result = item_class
            alert_conf = item_conf
        elif item_frame == longest_frame:
            item_index = alert_list.index(map[item_class])
            longest_frame_index = alert_list.index(map[alert_result])
            if item_index > longest_frame_index:
                alert_result = item_class
                alert_conf = item_conf
    return alert_result, longest_frame, alert_conf


def output_alert(alert_result, longest_frame, alert_conf, map):
    average_alert_conf = round(alert_conf / longest_frame, 2)
    warning_status = True
    alert_img_text = f'alert:{map[alert_result]}, conf:{average_alert_conf} last more than {alert_time}s'
    return warning_status, alert_img_text


if __name__ == '__main__':

    model_path = 'weights_gray'
# 配置模型路径 model loading path
    index_to_class = {0: 'safe_driving', 1: 'eating', 2: 'drinking',
                      3: 'smoking', 4: 'phone_interaction', 5: 'other_activity'}

    class_to_index = {v: k for k, v in index_to_class.items()}

    alert_list = ['smoking', 'drinking', 'eating', 'phone_interaction']
    safe_mode = 0  # 默认安全驾驶是 分类 0, safe driving is 0 by default

    video_name = 'test_video.avi'
# video name 需要更改, video name should be changed

    cap = cv2.VideoCapture(video_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'frame per second is {fps}')

    if fps == 0:
        sys.exit('fps should not be 0, exit program')

    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f'num frame is {num_frames}')

    video_length = round(num_frames / fps)
    print(f'video time length is {video_length}s')

    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f'frame height is {frame_height}. width is {frame_width}')

    reset_time = 2
    alert_time = 10
    buffer_time = 30
    print(f'reset time range is {reset_time}s, alert time range is {alert_time}s, buffer time range is {buffer_time}s')

    reset_frame = reset_time * fps
    alert_frame = alert_time * fps
    buffer_frame = buffer_time * fps
    print(f'{reset_frame} frames in reset time, {alert_frame} frames in alert time, {buffer_frame} frames in buffer time')

    buffer_list = []  # buffer queue for 30s accumulate

    safe_mode_buffer = 0
    state_previous = 0  # 初始化状态为安全驾驶 0, initial state is safe driving
    conf_threshold = 0.0
    warning_status = False # initial no warning

    model = load_model(model_path)
    dummy_img = cv2.imread('dummy_img.jpg')
    _ = model.predict(dummy_img[np.newaxis, ...])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_video = cv2.VideoWriter('output_func4.mp4', fourcc, fps, (640, 480), False)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print('read video failed or complete, break')
            break

        frame_now = cap.get(cv2.CAP_PROP_POS_FRAMES)

        time_start = time.time()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(frame, (224, 224))
        gray_img = np.repeat(gray_img[..., np.newaxis], 3, -1)

        gray_img_infer = model.predict(gray_img[np.newaxis, ...])
        gray_img_pred = np.argmax(gray_img_infer)
        conf = gray_img_infer[0][gray_img_pred]

        time_end = time.time()
        time_cost = round(time_end - time_start, 3)

        state_now = gray_img_pred
        output_text = f'current frame is {frame_now}, infer result is {map[gray_img_pred]}, infer time is {time_cost}'
        print(output_text)

        img_text = f'P: {index_to_class[gray_img_pred]}, conf: {round(conf, 2)}'

        # TODO:是否要添加置信度过滤 相当于跳过置信度低的一帧图片
        if conf >= conf_threshold:
            buffer_list.append((state_now, conf))
            if len(buffer_list) == buffer_frame:
                buffer_list = buffer_list[1:]
            # 建立与buffer_list对应的buffer
            buffer = {}  # buffer for sort out longest time and judge >= 10s
            for (item_class, item_conf) in buffer_list:
                if item_class not in buffer:
                    buffer[item_class] = (1, item_conf)
                else:
                    buffer[item_class][0] += 1
                    buffer[item_class][1] += conf

            if state_now != safe_mode:
                safe_mode_buffer = 0
                # alert 模块
                # 选取>=10s 最大时长动作发出alert
                # 如果<10s, 累加alert 动作时长， 看是否>=10s
                # 如果出现相同时长，按mobile phone > eating > drinking> smoking 优先级选取

                multi_activity_buffer = {k:v for k,v in buffer.items() if index_to_class[k] in alert_list}
                if len(multi_activity_buffer) != 0:
                    max_time = max([item[0] for item in multi_activity_buffer.values()])
                if max_time >= alert_frame:
                    alert_result, longest_frame, alert_conf = sort_buffer(multi_activity_buffer, class_to_index['smoking'], index_to_class)
                    # initial input lowest priority class to sort_buffer
                    warning_status, alert_img_text = output_alert(alert_result, longest_frame, alert_conf, index_to_class)
                    print(alert_img_text + ' ' + output_text)
                else:
                    total_time = sum([item[0] for item in multi_activity_buffer.values()])
                    if total_time >= alert_frame:
                        alert_result, longest_frame, alert_conf = sort_buffer(multi_activity_buffer, class_to_index['smoking'], index_to_class)
                        # initial input lowest priority class to sort_buffer
                        warning_status, alert_img_text = output_alert(alert_result, longest_frame, alert_conf, index_to_class)
                        print(alert_img_text + ' ' + output_text)
            else:
                if state_previous == safe_mode:
                    safe_mode_buffer += 1
                else:
                    safe_mode_buffer = 1
                if safe_mode_buffer >= reset_frame:
                    if warning_status is True:
                        warning_status = False
                        release_message = f'safe driving mode last for 2s, release warning'
                        img_text = release_message + img_text
                        print(release_message + ' ' + output_text)
                    safe_mode_buffer -= 1  # 防止一直累加 上溢出
            state_previous = state_now

        cv2.putText(frame, img_text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if warning_status is True:
            cv2.putText(frame, alert_img_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        save_video.write(frame)

    cap.release()
    save_video.release()

print('pred video is saved')








