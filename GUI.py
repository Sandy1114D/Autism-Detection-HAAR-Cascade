import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import os

# Create main window
root = tk.Tk()
root.title("Autism Face Recognition")
root.geometry("800x500")
root.resizable(False, False)

# Load background image
try:
    bg_image = Image.open("Autism.png").resize((800, 500))
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    messagebox.showerror("Error", f"Could not load background image:\n{e}")
    root.destroy()
    exit()

# Function to run external scripts
def run_script(script_name):
    try:
        subprocess.Popen(["python", script_name], shell=True)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {script_name}:\n{e}")

# Buttons with styling
button_font = ("Helvetica", 14, "bold")
button_bg = "#1976D2"
button_fg = "#FFFFFF"
button_width = 20

tk.Button(root, text="Capture Faces", font=button_font,
          bg=button_bg, fg=button_fg, width=button_width,
          command=lambda: run_script("gather_selfies.py")).place(x=280, y=150)

tk.Button(root, text="train Recognizer", font=button_font,
          bg=button_bg, fg=button_fg, width=button_width,
          command=lambda: run_script("train_recognizer.py")).place(x=280, y=220)

tk.Button(root, text="Run Recognition", font=button_font,
          bg=button_bg, fg=button_fg, width=button_width,
          command=lambda: run_script("Autism.py")).place(x=280, y=290)

# Start the GUI event loop
root.mainloop()
