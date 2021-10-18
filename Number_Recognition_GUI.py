import tkinter as tk
import Perceptron
import numpy as np


# ------------------------------
def recognition():
    number = list(number_entry.get())
    for i in range(len(number)):
        number[i] = int(number[i])
    result_label['text'] = str(my_model.predict(number))


# main function
np.random.seed(1)

config = {'path': 'Dataset/Number.txt',
          'learning_rate': 0.8,
          'check_packet_frequency': 30,
          'epoch': 600,
          'error_rate': 0.01}

dataset = Perceptron.Dataset(config['path'], True)
my_model = Perceptron.Model(dataset.feature, dataset.class_type, config['learning_rate'])
my_model = Perceptron.train_model(config, dataset, my_model)
my_model.to_best()
# ------------------------------
window = tk.Tk()
window.title('Number Recognition')
window.geometry('300x300')
window.resizable(False, False)

left_part = tk.Frame(window, width=150, height=300)
left_part.pack(side=tk.LEFT)
right_part = tk.Frame(window, width=150, height=300, bg='white')
right_part.pack(side=tk.LEFT)

number_label = tk.Label(left_part, text='Input Number')
number_label.pack()
number_entry = tk.Entry(left_part)
number_entry.pack()

run_btn = tk.Button(left_part, text='run', command=recognition)
run_btn.pack()

predict_label = tk.Label(right_part, text='Result')
predict_label.place(x=65, y=130)
result_label = tk.Label(right_part, text='')
result_label.place(x=70, y=150)

window.mainloop()
