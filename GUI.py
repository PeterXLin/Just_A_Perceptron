import tkinter as tk

window = tk.Tk()

window.title('Have Fun with Perceptron')
window.geometry('1200x700')
window.resizable(False, False)
# window.configure(background='white')


def get_configure_and_run():
    lr = float(lr_entry.get())
    result_label.config(text=lr)


left_part = tk.Frame(window, width=200, height=700)
left_part.pack(side=tk.LEFT)
right_part = tk.Frame(window, width=1000, height=700)
right_part.pack(side=tk.RIGHT)

# window object and layout
header_label = tk.Label(left_part, text='Perceptron Playground')
header_label.pack(side=tk.TOP)

lr_frame = tk.Frame(left_part)
lr_frame.pack(side=tk.TOP)
lr_label = tk.Label(lr_frame, text='Learning Rate')
lr_label.pack(side=tk.LEFT)
lr_entry = tk.Entry(lr_frame)
lr_entry.pack(side=tk.LEFT)


run_btn = tk.Button(left_part, text='run', command=get_configure_and_run)
run_btn.pack()
# TODO: set terminate condition

# TODO: 顯示訓練結果(包括訓練辨識率、測試辨識率、鍵結值等)
result_label = tk.Label(right_part)
result_label.pack()

# TODO: 二維資料能顯示資料點於二維座標的位置，並依照分群結果以不同顏色或符號表示。

window.mainloop()




