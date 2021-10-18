import tkinter as tk
from tkinter import ttk
import Perceptron
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog as fd


# -------------------------- functions --------------------------
def get_configure_and_run():
    get_configure()
    dataset = Perceptron.Dataset(config['path'])
    my_model = Perceptron.Model(dataset.feature, dataset.class_type, config['learning_rate'])
    my_model = Perceptron.train_model(config, dataset, my_model)
    my_model.to_best()
    # draw 2-d data distribution map
    fig1.clear()
    fig2.clear()
    fig3.clear()
    if dataset.feature == 2:
        draw_2d_data_distribution_map_and_prediction_line(np.concatenate(
            (dataset.training_dataset, dataset.validation_dataset), axis=0), 1, data_only=True)
        draw_2d_data_distribution_map_and_prediction_line(dataset.training_dataset,
                                                          2, False, my_model.best_neuron_list)
        draw_2d_data_distribution_map_and_prediction_line(dataset.validation_dataset,
                                                          3, False, my_model.best_neuron_list)
    elif dataset.feature == 3:
        draw_3d_data_distribution_map_and_prediction_line(np.concatenate(
            (dataset.training_dataset, dataset.validation_dataset), axis=0), 1, data_only=True)
        draw_3d_data_distribution_map_and_prediction_line(dataset.training_dataset,
                                                          2, False, my_model.best_neuron_list)
        draw_3d_data_distribution_map_and_prediction_line(dataset.validation_dataset,
                                                          3, False, my_model.best_neuron_list)

    training_label['text'] = str(get_correct_rate(dataset.training_dataset, my_model))
    testing_label['text'] = str(get_correct_rate(dataset.validation_dataset, my_model))
    weight_label['text'] = str(my_model.best_neuron_list)


def get_configure():
    try:
        config['learning_rate'] = float(lr_entry.get())
        config['check_packet_frequency'] = int(check_entry.get())
        config['epoch'] = int(epoch_entry.get())
        config['error_rate'] = float(error_entry.get())
        error_message['text'] = ''
    except ValueError:
        error_message['text'] = 'Please enter a number'


def draw_2d_data_distribution_map_and_prediction_line(dataset: "np.ndarry", position: int, data_only=True,
                                                      all_weights: list = None):
    """draw figure and show it on window"""
    if position == 1:
        fig = fig1
        canvas = canvas1
    elif position == 2:
        fig = fig2
        canvas = canvas2
    else:
        fig = fig3
        canvas = canvas3
    fig.clear()
    plot1 = fig.add_subplot(111)

    x_1, x_2, classes = separate_2d_data_by_class(dataset)
    color_list = ["orange", "blue", "brown", "red",
                  "grey", "yellow", "green", "pink"]
    for i in range(len(x_1)):
        plot1.scatter(x_1[i], x_2[i], c=color_list[i % 8], s=5)

    if not data_only:
        x_range = get_x_range(dataset)
        y_range = get_x_range(dataset, 1)
        count = 1
        for weight in all_weights:
            if weight[1] == 0:
                x = x_range
                y = -weight[0] / weight[1]
                plot1.plot(x, y, c=color_list[-count])
            elif weight[2] == 0:
                y = y_range
                x = -weight[0] / weight[2]
                plot1.plot(x, y, c=color_list[-count])
            else:
                x = x_range
                y = -(-weight[0] + weight[1] * x) / weight[2]
                plot1.plot(x, y, c=color_list[-count])
            count = count + 1
    # other setting
    plot1.axhline(y=0)
    plot1.axvline(x=0)
    canvas.draw()
    canvas.get_tk_widget().pack()


def draw_3d_data_distribution_map_and_prediction_line(dataset: "np.ndarry", position: int, data_only=True,
                                                      all_weights: list = None):
    """draw figure and show it on window"""
    if position == 1:
        fig = fig1
        canvas = canvas1
    elif position == 2:
        fig = fig2
        canvas = canvas2
    else:
        fig = fig3
        canvas = canvas3
    plot1 = fig.add_subplot(111, projection='3d')

    x_1, x_2, x_3, classes = separate_3d_data_by_class(dataset)
    color_list = ["orange", "blue", "brown", "red",
                  "grey", "yellow", "green", "pink"]

    for i in range(len(x_1)):
        plot1.scatter(x_1[i], x_2[i], x_3[i], c=color_list[i % 8], s=5)

    if not data_only:
        x = get_x_range(dataset)
        y = get_x_range(dataset, 1)
        if len(y) > len(x):
            x = y
        count = 1
        for weight in all_weights:
            z = -(-weight[0] + weight[1] * x + weight[2] * x) / weight[3]
            plot1.plot(x, x, z, c=color_list[-count])
            count = count + 1
    # other setting
    canvas.draw()
    canvas.get_tk_widget().pack()


def separate_2d_data_by_class(dataset: np.ndarray):
    x_1 = list()
    x_2 = list()
    class_list = Perceptron.get_class_list(dataset)
    for i in range(len(class_list)):
        x_1.append(list())
        x_2.append(list())
    for d in dataset:
        type_index = int(class_list.index(d[-1]))
        x_1[type_index].append(d[0])
        x_2[type_index].append(d[1])
    return x_1, x_2, len(class_list)


def separate_3d_data_by_class(dataset: np.ndarray):
    x_1 = list()
    x_2 = list()
    x_3 = list()
    class_list = Perceptron.get_class_list(dataset)
    for i in range(len(class_list)):
        x_1.append(list())
        x_2.append(list())
        x_3.append(list())
    for d in dataset:
        type_index = int(class_list.index(d[-1]))
        x_1[type_index].append(d[0])
        x_2[type_index].append(d[1])
        x_3[type_index].append(d[2])
    return x_1, x_2, x_3, len(class_list)


def get_x_range(dataset: np.ndarray, feature_index: int = 0):
    x1_min = np.amin(dataset, axis=0)[feature_index]
    x1_max = np.amax(dataset, axis=0)[feature_index]
    return np.arange(x1_min, x1_max, 0.1)


def get_correct_rate(dataset: np.ndarray, model: Perceptron.Model):
    data_amount = len(dataset)
    error_amount = 0
    for data in dataset:
        if model.predict(data) != data[-1]:
            error_amount = error_amount + 1
    return (data_amount - error_amount) / data_amount


def select_file():
    filetypes = (
        ('text files', '*.txt'),
    )
    config['path'] = fd.askopenfilename(
        title='Open a file',
        initialdir='./Dataset',
        filetypes=filetypes)


# ------------------------ Config --------------------------
np.random.seed(1)

config = {'path': 'Dataset/2Hcircle1.txt',
          'learning_rate': 0.8,
          'check_packet_frequency': 30,
          'epoch': 600,
          'error_rate': 0.01,}
# --------------------------- GUI --------------------------------
window = tk.Tk()
window.title('Have Fun with Perceptron')
window.geometry('1000x750')
window.resizable(False, False)
# window.configure(background='white')

left_part = tk.Frame(window, width=200, height=750)
left_part.pack(side=tk.LEFT)
middle_part = tk.Frame(window, width=350, height=750, bg='white')
middle_part.pack(side=tk.LEFT)
middle_up = tk.Frame(middle_part, width=350, height=250, bg='white')
middle_up.pack()
middle_middle = tk.Frame(middle_part, width=350, height=250, bg='white')
middle_middle.pack()
middle_down = tk.Frame(middle_part, width=350, height=250, bg='white')
middle_down.pack()
right_part = tk.Frame(window, width=450, height=750)
right_part.pack(side=tk.LEFT)

# -------------------- Left Part Object -----------------------------
header_label = tk.Label(left_part, text='Perceptron Playground', font=("Courier", 16))
header_label.pack(side=tk.TOP)

file_frame = tk.Frame(left_part)
file_frame.pack()
file_label = tk.Label(file_frame, text='File')
file_label.pack(side=tk.LEFT)
open_button = ttk.Button(file_frame, text='choose a File', command=select_file)
open_button.pack(side=tk.LEFT)

# file_select = ttk.Combobox(file_frame, values=file_list, state='readonly')
# file_select.pack(side=tk.LEFT)

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
fig1 = Figure(figsize=(3, 2.5), dpi=100)
fig2 = Figure(figsize=(3, 2.5), dpi=100)
fig3 = Figure(figsize=(3, 2.5), dpi=100)
canvas1 = FigureCanvasTkAgg(fig1, master=middle_up)
canvas1.draw()
canvas1.get_tk_widget().pack(side=tk.TOP, fill='y')
canvas2 = FigureCanvasTkAgg(fig2, master=middle_middle)
canvas2.draw()
canvas2.get_tk_widget().pack(side=tk.TOP, fill='y')
canvas3 = FigureCanvasTkAgg(fig3, master=middle_down)
canvas3.draw()
canvas3.get_tk_widget().pack(side=tk.TOP, fill='y')
# ---------------------------------------------------------------------

training_text = tk.Label(right_part, text='Training Data Correct Rate')
training_text.pack()
training_label = tk.Label(right_part, text='', wraplength=400)
training_label.pack()
testing_text = tk.Label(right_part, text='Testing Data Correct Rate')
testing_text.pack()
testing_label = tk.Label(right_part, text='', wraplength=400)
testing_label.pack()
weight_text = tk.Label(right_part, text='Weights')
weight_text.pack()
weight_label = tk.Label(right_part, text='', wraplength=400)
weight_label.pack()

window.mainloop()
