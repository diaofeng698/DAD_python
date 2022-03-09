import numpy as np
import cv2
import sys
import time


if __name__ == '__main__':

    model_path = 'weights_gray'
# 配置模型路径 model loading path
    index_to_class = {0: 'safe_driving', 1: 'eating', 2: 'drinking',
                      3: 'smoking', 4: 'phone_interaction', 5: 'other_activity'}

    class_to_index = {'sf': 0, 'ea': 1, 'dr': 2, "sm": 3, 'pi': 4, 'oa': 5}

    alert_list = ['smoking', 'drinking', 'eating', 'phone_interaction']
    safe_mode = 0  # 默认安全驾驶是 分类 0, safe driving is 0 by default

    fps = 5

    if fps == 0:
        sys.exit('fps should not be 0, exit program')

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

    # 构造测试数据队列
    save_test_route = './function_test_result/test_result4.txt'
# 测试结果保存文件 每次测试需要提供新的文件名
    time_dict = {'sf': 5, 'ea': 4, 'dr': 3, 'sm': 2, 'pi': 11}
    time_dict['oa'] = buffer_time - sum(time_dict.values())
    print(f'activity last time dict is {time_dict}')

    total_test_time = 60  # 总测试时间60s

    test_queue = []
    record = ''
    for _ in range((total_test_time // buffer_time)):
        for item_class, item_time in time_dict.items():
            test_queue.extend([class_to_index[item_class]] * item_time * fps)
            record += f'{item_class} lasts for {item_time}s, '
        print(f'in {buffer_time}, {record}')
        record = ''
    frame_index = 1
    for frame_now in test_queue:
        state_now = frame_now
        conf = 1
        output_text = f'current frame No.is {frame_index}, frame is {index_to_class[state_now]}'
        #print(output_text)

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
                            if index_to_class[item_class] in alert_list:
                                if item_frame > longest_frame:
                                    longest_frame = item_frame
                                    alert_result = index_to_class[item_class]
                                    alert_conf = item_conf
                                elif item_frame == longest_frame:
                                    item_index = alert_list.index(index_to_class[item_class])
                                    longest_frame_index = alert_list.index(index_to_class[item_class])
                                    if item_index >= longest_frame_index:
                                        alert_result = index_to_class[item_class]
                                        alert_conf = item_conf
                    if longest_frame != 0:
                        average_alert_conf = round(alert_conf / longest_frame, 2)
                        warning_status = True
                        alert_img_text = f'alert:{alert_result}, conf:{average_alert_conf} last more than {alert_time}s'
                        #print(alert_img_text + ' ' + output_text)

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
                            reset_message = 'safe driving mode last for 2s, release warning'
                            output_text = reset_message + ', ' + output_text
                            # print(reset_message)
                        safe_mode_buffer -= 1  # 防止一直累加 上溢出
                        buffer = {k: [0, 0] for k in range(classes)}
                        buffer_list = []
                        counter = 1  # reset 回到30s 内重新累加
                    state_previous = state_now

            counter += 1

        else:  # 第31s...，滑动窗口为1个frame, 31s, sliding window every 1 frame
            if state_now == safe_mode:
                # 解除warning 判断模块, release warning module
                if conf >= conf_threshold:
                    if state_previous == safe_mode:
                        safe_mode_buffer += 1
                    else:
                        safe_mode_buffer = 1
                    if safe_mode_buffer >= reset_frame:
                        if warning_status is True:
                            warning_status = False
                            reset_message = 'safe driving mode last for 2s, release warning'
                            output_text = reset_message + ', ' + output_text
                            #print(reset_message)
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
                            if index_to_class[item_class] in alert_list:
                                if item_frame > longest_frame:
                                    longest_frame = item_frame
                                    alert_result = index_to_class[item_class]
                                    alert_conf = item_conf
                                elif item_frame == longest_frame:
                                    item_index = alert_list.index(index_to_class[item_class])
                                    longest_frame_index = alert_list.index(index_to_class[item_class])
                                    if item_index >= longest_frame_index:
                                        alert_result = index_to_class[item_class]
                                        alert_conf = item_conf
                    if longest_frame != 0:
                        average_alert_conf = round(alert_conf / longest_frame, 2)
                        warning_status = True
                        alert_img_text = f'alert:{alert_result}, conf:{average_alert_conf} last more than {alert_time}s'
                        #print(alert_img_text + ' ' + output_text)

                    state_previous = state_now

        with open(save_test_route, 'a') as file:
            if warning_status is True:
                file.write(alert_img_text + ', ' + output_text + '\n')
            else:
                file.write(output_text + '\n')

        frame_index += 1
    print(f'test in finished, result is saved at {save_test_route}')







