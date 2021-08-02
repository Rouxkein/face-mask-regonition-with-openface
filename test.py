import os
from tkinter.filedialog import askopenfilename
import tkinter.messagebox
import tkinter as tk

# Dir select
def create_file():
    path_ = 'C:/Users/ADMIN/PycharmProjects/face-recognition-using-deep-learning-master/dataset'
    path.set(path_)
    print("folder_name: ", folder.get())
    print("path_name: ", path.get())
    dirs = os.path.join(path.get(), folder.get())
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        tkinter.messagebox.showinfo('Tips:','Folder name created successfully!')
    else:
        tkinter.messagebox.showerror('Tips','The folder name exists, please change it')

root = tk.Tk()
root.title('Create folder')
root.geometry('400x380')

path = tk.StringVar()   # Receiving user's file_path selection
folder = tk.StringVar() # Receiving user's folder_name selection

# tk.Label(root,text = "Target path:").place(x=50, y= 250)
# tk.Entry(root, textvariable = path).place(x=110, y= 250)
# tk.Button(root, text = "Path select: ", command = selectPath).place(x=265, y= 250)
tk.Label(root,text = "Folder name:").place(x=50, y= 300)
tk.Entry(root,textvariable = folder).place(x=110, y= 300)
tk.Button(root, text = "Submit: ", command = create_file).place(x=265, y= 300)

root.mainloop()