import os


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
    alert_img_text = f'alert:{map[alert_result]}, conf:{average_alert_conf} last longest time in 50 frames'
    return warning_status, alert_img_text


if __name__ == '__main__':
    root = os.getcwd()
    save_test_name = 'test_result7.txt'
# 更改结果保存文件名称
    save_test_folder = 'function_test_result'
    save_test_path = os.path.join(root, save_test_folder)
    if not os.path.exists(save_test_path):
        os.mkdir(save_test_path)
    save_test_route = os.path.join(save_test_path, save_test_name)

    model_path = 'weights_gray'
# 配置模型路径 model loading path
    index_to_class = {0: 'safe_driving', 1: 'eating', 2: 'drinking',
                      3: 'smoking', 4: 'phone_interaction', 5: 'other_activity'}

    class_to_index = {v:k for k, v in index_to_class.items()}

    alert_list = ['smoking', 'drinking', 'eating', 'phone_interaction']
    safe_mode = 0  # 默认安全驾驶是 分类 0, safe driving is 0 by default
    fps = 5
# 默认测试FPS 为 5
    reset_time = 2
    alert_time = 10
    buffer_time = 30
    print(f'reset time range is {reset_time}s, alert time range is {alert_time}s, buffer time range is {buffer_time}s')

    reset_frame = reset_time * fps
    alert_frame = alert_time * fps
    buffer_frame = buffer_time * fps
    print(f'{reset_frame} frames in reset time, {alert_frame} frames in alert time, {buffer_frame} frames in buffer time')

    buffer_list = []

    safe_mode_buffer = 0
    state_previous = 0  # 初始化状态为安全驾驶 0, initial state is safe driving
    conf_threshold = 0.0
    warning_status = False

    # 构造测试数据队列
    time_list = [('safe_driving', 5), ('eating', 4), ('drinking', 3), ('smoking', 2),
                 ('phone_interaction', 11), ('other_activity', 60)]
    print(f'activity last time dict is {time_list}')

    with open(save_test_route, 'a') as file:
        file.write(str(time_list) + '\n')

    test_queue = []
    record = ''

    for item in time_list:
        test_queue.extend([class_to_index[item[0]]] * item[1] * fps)
        record += f'{item[0]} lasts for {item[1]}s, '
    record_message = f'in {sum([item[1] for item in time_list])}, {record}'
    print(record_message)
    with open(save_test_route, 'a') as file:
        file.write(record_message + '\n')

    frame_index = 1
    for frame_now in test_queue:
        state_now = frame_now
        conf = 1  # conf 其实可以随机生成一个值 在(0, 1) 之间
        output_text = f'current frame No.is {frame_index}, frame is {index_to_class[state_now]}'
        print(output_text)
        # TODO: 是否要添加置信度过滤 相当于跳过置信度低的一帧图片
        if conf >= conf_threshold:
            buffer_list.append((state_now, conf))
            if len(buffer_list) == buffer_frame:
                buffer_list = buffer_list[1:]
            # 建立与buffer_list对应的buffer
            buffer = {}
            for (item_class, item_conf) in buffer_list:
                if item_class not in buffer:
                    buffer[item_class] = [1, item_conf]
                else:
                    buffer[item_class][0] += 1
                    buffer[item_class][1] += conf

            if state_now != safe_mode:
                safe_mode_buffer = 0
                # alert 模块
                # 选取>=10s 最大时长动作发出alert
                # 如果<10s, 累加alert 动作时长， 看是否>=10s
                # 如果出现相同时长，按mobile phone > eating > drinking> smoking 优先级选取

                multi_activity_buffer = {k: v for k, v in buffer.items() if index_to_class[k] in alert_list}
                if len(multi_activity_buffer) != 0:
                    max_time = max([item[0] for item in multi_activity_buffer.values()])
                if max_time >= alert_frame:
                    print('OK')
                    alert_result, longest_frame, alert_conf = sort_buffer(multi_activity_buffer, class_to_index['smoking'], index_to_class)
                    # initial input lowest priority class to sort_buffer
                    warning_status, alert_img_text = output_alert(alert_result, longest_frame, alert_conf, index_to_class)
                    save_text = 'single activity' + ' ' + alert_img_text
                else:
                    total_time = sum([item[0] for item in multi_activity_buffer.values()])
                    if total_time >= alert_frame:
                        print('ok')
                        alert_result, longest_frame, alert_conf = sort_buffer(multi_activity_buffer, class_to_index['smoking'], index_to_class)
                        # initial input lowest priority class to sort_buffer
                        warning_status, alert_img_text = output_alert(alert_result, longest_frame, alert_conf, index_to_class)
                        save_text = 'multi  activity' + ' ' + alert_img_text
            else:
                if state_previous == safe_mode:
                    safe_mode_buffer += 1
                else:
                    safe_mode_buffer = 1
                if safe_mode_buffer >= reset_frame:
                    if warning_status is True:
                        warning_status = False
                        release_message = f'safe driving mode last for 10 frames, release warning'
                        output_text = release_message + ' ' + output_text
                        #print(save_text)
                    safe_mode_buffer -= 1  # 防止一直累加 上溢出
            state_previous = state_now

        with open(save_test_route, 'a') as file:
            if warning_status is True:
                file.write(save_text + ' ' + output_text + '\n')
            else:
                file.write(output_text + '\n')

        frame_index += 1
    print(f'test in finished, result is saved at {save_test_route}')







