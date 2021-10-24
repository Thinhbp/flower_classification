import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model
import easygui
model = load_model('C:\\Users\\Admin\\Desktop\\pythonProject\\Flower\\vggmodel.h5')
model.load_weights('C:\\Users\\Admin\\Desktop\\pythonProject\\Flower\\weights-03-0.77.hdf5')
classes = {
    0:'its a daisy',
    1:'its a dandelion',
    2:'its a rose',
    3:'its a sunflowers',
    4:'its a tulip'
}
top=tk.Tk()
top.geometry('800x600')
top.title('Flower Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((128,128))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image/255
    prediction = np.argmax(model.predict(image))
    type_flower = classes[prediction]
    easygui.msgbox(type_flower, title="Result")

    label.configure(foreground='#011638', text=type_flower)

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify flower",
    command=lambda: classify(file_path),
    padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
        (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Flower Classification",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()