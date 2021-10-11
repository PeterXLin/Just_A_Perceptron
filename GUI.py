import tkinter as tk
from tkinter import ttk
import Perceptron
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


# -------------------------- functions --------------------------
def get_configure_and_run():
    # get_configure()
    dataset = Perceptron.Dataset(config['path'], config['classes'])
    my_model = Perceptron.Model(dataset.feature, config['classes'], config['learning_rate'])
    my_model = Perceptron.train_model(config, dataset, my_model)
    if config['classes'] == 2:
        get_data_distribution_map_and_prediction_line(np.concatenate(
            (dataset.training_dataset, dataset.validation_dataset), axis=0),
            111, data_only=True)
        get_data_distribution_map_and_prediction_line(dataset.training_dataset,
                                                      121, False, my_model.best_neuron_list[0])
        get_data_distribution_map_and_prediction_line(dataset.validation_dataset,
                                                      121, False, my_model.best_neuron_list[0])


def get_configure():
    file_name = file_select.get()
    config['path'] = 'Dataset/' + file_name
    if more_than_two_dict.get(file_name) is None:
        config['classes'] = 2
    else:
        config['classes'] = more_than_two_dict[file_name]

    try:
        config['learning_rate'] = float(lr_entry.get())
        config['check_packet_frequency'] = int(check_entry.get())
        config['epoch'] = int(epoch_entry.get())
        config['error_rate'] = float(error_entry.get())
        error_message['text'] = ''
    except ValueError:
        error_message['text'] = 'Please enter a number'


def get_data_distribution_map_and_prediction_line(dataset: 'np.ndarray', position: int, data_only=True,
                                                  weight: list = None):
    """draw figure and show it on window"""
    fig = Figure(figsize=(3, 3), dpi=100)
    plot1 = fig.add_subplot(position)
    x_1, y_1, x_2, y_2 = separate_data_by_class(dataset)
    plot1.scatter(x_1, y_1, c='red', s=5)
    plot1.scatter(x_2, y_2, c='green', s=5)
    if not data_only:
        x = get_x_range(dataset)
        y = -(-weight[0] + weight[1] * x) / weight[2]
        plot1.plot(x, y, c='purple')
    # other setting
    plot1.axhline(y=0)
    plot1.axvline(x=0)
    canvas = FigureCanvasTkAgg(fig, master=middle_part)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
    # toolbar = NavigationToolbar2Tk(canvas, right_part)
    # toolbar.update()
    # canvas.get_tk_widget().pack()


def separate_data_by_class(dataset: np.ndarray):
    x_1_class_1 = list()
    x_1_class_2 = list()
    x_2_class_1 = list()
    x_2_class_2 = list()
    for d in dataset:
        if d[2] == 1:
            x_1_class_1.append(d[0])
            x_2_class_1.append(d[1])
        elif d[2] == 2:
            x_1_class_2.append(d[0])
            x_2_class_2.append(d[1])
    return x_1_class_1, x_2_class_1, x_1_class_2, x_2_class_2


def get_x_range(dataset: np.ndarray):
    x1_min = np.amin(dataset, axis=0)[0]
    x1_max = np.amax(dataset, axis=0)[0]
    return np.arange(x1_min, x1_max, 0.1)


# ------------------------ Config --------------------------
np.random.seed(1)
file_list = ['perceptron1.txt', 'perceptron2.txt', '2Ccircle1.txt',
             '2Circle1.txt', '2Circle2.txt', '2CloseS.txt', '2CloseS2.txt',
             '2CloseS3.txt', '2cring.txt', '2CS.txt', '2Hcircle1.txt', '2ring.txt']

more_than_two_dict = {}

config = {'path': 'Dataset/2Hcircle1.txt',
          'classes': 2,
          'learning_rate': 0.8,
          'check_packet_frequency': 30,
          'epoch': 600,
          'error_rate': 0.1}
# --------------------------- GUI --------------------------------
window = tk.Tk()
window.title('Have Fun with Perceptron')
window.geometry('1200x700')
window.resizable(False, False)
# window.configure(background='white')

left_part = tk.Frame(window, width=200, height=700)
left_part.pack(side=tk.LEFT)
middle_part = tk.Frame(window, width=1000, height=700)
middle_part.pack(side=tk.LEFT)
# right_part = tk.Frame(window, width=400, height=700)
# right_part.pack(side=tk.LEFT)

# -------------------- Left Part Object -----------------------------
header_label = tk.Label(left_part, text='Perceptron Playground')
header_label.pack(side=tk.TOP)

file_frame = tk.Frame(left_part)
file_frame.pack()
file_label = tk.Label(file_frame, text='File')
file_label.pack(side=tk.LEFT)
file_select = ttk.Combobox(file_frame, values=file_list, state='readonly')
file_select.pack(side=tk.LEFT)

lr_frame = tk.Frame(left_part)
lr_frame.pack()
lr_label = tk.Label(lr_frame, text='Learning Rate')
lr_label.pack(side=tk.LEFT)
lr_entry = tk.Entry(lr_frame)
lr_entry.pack(side=tk.LEFT)

check_frame = tk.Frame(left_part)
check_frame.pack()
check_label = tk.Label(check_frame, text='Check Pocket Frequency')
check_label.pack(side=tk.LEFT)
check_entry = tk.Entry(check_frame)
check_entry.pack(side=tk.LEFT)

epoch_frame = tk.Frame(left_part)
epoch_frame.pack()
epoch_label = tk.Label(epoch_frame, text='Epoch')
epoch_label.pack(side=tk.LEFT)
epoch_entry = tk.Entry(epoch_frame)
epoch_entry.pack(side=tk.LEFT)

error_frame = tk.Frame(left_part)
error_frame.pack()
error_label = tk.Label(error_frame, text='Error Rate')
error_label.pack(side=tk.LEFT)
error_entry = tk.Entry(error_frame)
error_entry.pack(side=tk.LEFT)

error_message = tk.Label(left_part, text='')
error_message.pack()

run_btn = tk.Button(left_part, text='run', command=get_configure_and_run)
run_btn.pack()
# --------------------------------------------------------------------
# TODO: 顯示訓練結果(包括訓練辨識率、測試辨識率、鍵結值等)

# TODO: 二維資料能顯示資料點於二維座標的位置，並依照分群結果以不同顏色或符號表示。

window.mainloop()
