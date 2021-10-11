import tkinter as tk
from tkinter import ttk
import Perceptron


# -------------------------- functions --------------------------
def get_configure_and_run():
    get_configure()


def get_configure():
    config['path'] = 'Dataset/' + file_select.get()
    try:
        config['learning_rate'] = float(lr_entry.get())
        config['check_packet_frequency'] = int(check_entry.get())
        config['epoch'] = int(epoch_entry.get())
        config['error_rate'] = float(error_entry.get())
        error_message['text'] = ''
    except ValueError:
        error_message['text'] = 'Please enter a number'


# ------------------------ Config --------------------------
file_list = ['perceptron1.txt', 'perceptron2.txt', '2Ccircle1.txt',
             '2Circle1.txt', '2Circle2.txt', '2CloseS.txt', '2CloseS2.txt',
             '2CloseS3.txt', '2cring.txt', '2CS.txt', '2Hcircle1.txt', '2ring.txt']

config = {'path': 'Dataset/2Hcircle1.txt',
          'classes': 2,
          'learning_rate': 0.8,
          'check_packet_frequency': 30,
          'epoch': 300,
          'error_rate': 0.1}
# -----------------------------------------------------------
window = tk.Tk()
window.title('Have Fun with Perceptron')
window.geometry('1200x700')
window.resizable(False, False)
# window.configure(background='white')

left_part = tk.Frame(window, width=200, height=700)
left_part.pack(side=tk.LEFT)
right_part = tk.Frame(window, width=1000, height=700)
right_part.pack(side=tk.RIGHT)

# -------------------- Left Part Object -----------------------------
header_label = tk.Label(left_part, text='Perceptron Playground')
header_label.pack(side=tk.TOP)

file_frame = tk.Frame(left_part)
file_frame.pack()
file_label = tk.Label(file_frame, text='File')
file_label.pack(side=tk.LEFT)
file_select = ttk.Combobox(file_frame, values=file_list)
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
# TODO: set terminate condition

# TODO: 顯示訓練結果(包括訓練辨識率、測試辨識率、鍵結值等)
result_label = tk.Label(right_part)
result_label.pack()

# TODO: 二維資料能顯示資料點於二維座標的位置，並依照分群結果以不同顏色或符號表示。

window.mainloop()




