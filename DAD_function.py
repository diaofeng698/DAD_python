import tensorflow as tf
import numpy as np
import cv2
import sys
import time


def load_model(path):
    return tf.keras.models.load_model(path)


if __name__ == '__main__':

    model_path = 'weights_gray'
# 配置模型路径 model loading path
    map_list = {0: 'safe_driving', 1: 'eating', 2: 'drinking',
                3: 'smoking', 4: 'phone_interaction', 5: 'other_activity'}

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

    classes = 6  # DAD classes number
    buffer = {k: [0, 0] for k in range(classes)}
    buffer_list = []
    print(f'buffer in {buffer_time}s is {buffer}')

    safe_mode_buffer = 0
    state_previous = 0  # 初始化状态为安全驾驶 0, initial state is safe driving
    conf_threshold = 0.5
    warning_status = False
    counter = 1  # buffer time counter

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

        img_text = f'P: {map_list[gray_img_pred]}, conf: {round(conf, 2)}'

        if counter <= buffer_frame:  # 1-12s, DAD: 1-30s
            if state_now != safe_mode:
                if conf >= conf_threshold:
                    safe_mode_buffer = 0
                    buffer[state_now][0] += 1
                    buffer[state_now][1] += conf
                    buffer_list.append((state_now, conf))
                    # alert 模块, alert module
                    # 选取最大时长动作发出alert, 如果出现相同时长，按phone_interaction > eating > drinking> smoking 优先级选取
                    # choose the longest time activity to alert, if there are two equal longest activities,
                    # sort by the priority phone_interaction > eating > drinking> smoking
                    longest_frame = 0
                    alert_result = 3  # 默认初始值为优先级最低的 'smoking', default value is the lowest priority 'smoking' class
                    alert_conf = 0
                    for item_class, (item_frame, item_conf) in buffer.items():
                        if item_frame >= alert_frame:
                            if map_list[item_class] in alert_list:
                                if item_frame > longest_frame:
                                    longest_frame = item_frame
                                    alert_result = map_list[item_class]
                                    alert_conf = item_conf
                                elif item_frame == longest_frame:
                                    item_index = alert_list.index(map_list[item_class])
                                    longest_frame_index = alert_list.index(map_list[item_class])
                                    if item_index >= longest_frame_index:
                                        alert_result = map_list[item_class]
                                        alert_conf = item_conf
                    if longest_frame != 0:
                        average_alert_conf = round(alert_conf / longest_frame, 2)
                        warning_status = True
                        alert_img_text = f'alert:{alert_result}, conf:{average_alert_conf} last more than {alert_time}s'
                        print(alert_img_text + ' ' + output_text)

                    state_previous = state_now
            else:
                # 解除warning 判断模块, release warning module
                if conf >= conf_threshold:
                    if state_previous == safe_mode:
                        safe_mode_buffer += 1
                    else:
                        safe_mode_buffer = 1
                    if safe_mode_buffer >= reset_frame:
                        if warning_status is True:
                            warning_status = False
                            print('safe driving mode last for 2s, release warning')
                        safe_mode_buffer -= 1  # 防止一直累加 上溢出
                        buffer = {k: [0, 0] for k in range(classes)}
                        buffer_list = []
                        counter = 1  # reset 回到30s 内重新累加
                    state_previous = state_now

            counter += 1

        else:  # 第31s...，滑动窗口为1个frame, 31s, sliding window every 1 frame
            if state_now == safe_mode:
                # 解除warning 判断模块, release warning module
                if conf >= conf_threshold :
                    if state_previous == safe_mode:
                        safe_mode_buffer += 1
                    else:
                        safe_mode_buffer = 1
                    if safe_mode_buffer >= reset_frame:
                        if warning_status is True:
                            warning_status = False
                            print('safe driving mode last for 2s, release warning')
                        safe_mode_buffer -= 1  # 防止一直累加 上溢出， in case overflow
                        buffer = {k: [0, 0] for k in range(classes)}
                        buffer_list = []
                        counter = 1  # reset 回到30s内重新循环，reset back to 30s loop
                    state_previous = state_now
            else:  # 如果第31s不是safe mode,没有被reset 就一直滑动下去, if 31s is not safe mode, continue sliding
                if conf >= conf_threshold:
                    safe_mode_buffer = 0
                    buffer_list.append((state_now, conf))
                    buffer_list = buffer_list[1:]  # buffer_list 首出 尾进， 滑动窗口为1, buffer list head out, end in
                    buffer = {k: [0, 0] for k in range(classes)}
                    for item in buffer_list:
                        item_class = item[0]
                        item_conf = item[1]
                        buffer[item_class][0] += 1
                        buffer[item_class][1] += item_conf
                    # alert 模块, alert module
                    # 选取最大时长动作发出alert, 如果出现相同时长，按phone_interaction > eating > drinking> smoking 优先级选取
                    # choose the longest time activity to alert, if there are two equal longest activities,
                    # sort by the priority phone_interaction > eating > drinking> smoking
                    longest_frame = 0
                    alert_result = 3  # 默认初始值为优先级最低的 'smoking', default value is the lowest priority 'smoking' class
                    alert_conf = 0
                    for item_class, (item_frame, item_conf) in buffer.items():
                        if item_frame >= alert_frame:
                            if map_list[item_class] in alert_list:
                                if item_frame > longest_frame:
                                    longest_frame = item_frame
                                    alert_result = map_list[item_class]
                                    alert_conf = item_conf
                                elif item_frame == longest_frame:
                                    item_index = alert_list.index(map_list[item_class])
                                    longest_frame_index = alert_list.index(map_list[item_class])
                                    if item_index >= longest_frame_index:
                                        alert_result = map_list[item_class]
                                        alert_conf = item_conf
                    if longest_frame != 0:
                        average_alert_conf = round(alert_conf / longest_frame, 2)
                        warning_status = True
                        alert_img_text = f'alert:{alert_result}, conf:{average_alert_conf} last more than {alert_time}s'
                        print(alert_img_text + ' ' + output_text)

                    state_previous = state_now

        cv2.putText(frame, img_text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if warning_status is True:
            cv2.putText(frame, alert_img_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        save_video.write(frame)

    cap.release()
    save_video.release()

    print('Activity detection video is saved')







