# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Tk-inter window implementations
"""

import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk


class MainFrame(ttk.Frame):

    def __init__(self, container):
        super().__init__(container)

        # variables to store path and dir
        self.error_msg = ''
        self.orig_img = None

        # set up grid layout
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=3)

        # initialize widgets
        self.__create_widgets()

    def __create_widgets(self):

        # (button) About
        self.btn_about = ttk.Button(self, text='About')
        self.btn_about['command'] = self.btn_about_clicked
        self.btn_about.grid(column=1, row=0, padx=5, pady=5, sticky=tk.E)

        # (button) Select Image File
        self.btn_select_img = tk.Button(self, text='Select Image', width=15, height=2, compound='c')
        self.btn_select_img['fg'] = 'dark green'
        self.btn_select_img['command'] = self.btn_select_img_clicked
        self.btn_select_img.grid(column=0, row=1, padx=5, pady=2, sticky=tk.EW)

        # (button) ChaosGT Algorithm
        self.btn_chaosgt = tk.Button(self, text='Proceed to Chaos GT', width=15, height=2, compound='c')
        self.btn_chaosgt['fg'] = 'dark green'
        self.btn_chaosgt['command'] = self.btn_chaosgt_clicked
        self.btn_chaosgt['state'] = 'disabled'
        self.btn_chaosgt.grid(column=0, row=2, padx=5, pady=2, sticky=tk.NW)

        # (label) Show status/error
        self.pgb_status = ttk.Progressbar(self, mode='indeterminate')
        # self.pgb_status.grid(column=0, row=3, padx=5, pady=5, sticky=tk.SW)
        self.pgb_status.grid_forget();  # hide temporarily

        self.lbl_msg = tk.Label(self, text='', fg='gray')
        # self.lbl_msg.grid(column=0, row=4, padx=5, pady=2, sticky=tk.NW)
        self.lbl_msg.grid_forget();  # hide temporarily

        # Image Placeholder
        pixel = tk.PhotoImage(width=1, height=1)
        self.lbl_image = tk.Label(self, image=pixel, width=300, height=300, bg='white')
        self.lbl_image.grid(column=1, row=1, rowspan=12, padx=5, pady=2, sticky=tk.EW)

    def btn_about_clicked(self):
        about_info = "Chaos-GT is an application for implemeting chaos engineering in nano-structures. "\
                     "The application was designed by Dickson Owuor and Nicholas Kotov."\
                     "\n\n\n" \
                     "Copyright (C) 2024, University of Michigan."\
                     "\n\n"
        messagebox.showinfo(
            title = 'About Chaos GT',
            message = about_info
        )

    def btn_select_img_clicked(self):

        # disable btn_select until image is loaded
        self.btn_select_img['state'] = 'disabled'
        self.btn_chaosgt['state'] = 'disabled'
        self.pgb_status.grid(column=0, row=4, padx=5, pady=1, sticky=tk.SW)
        self.lbl_msg.grid(column=0, row=5, padx=5, pady=1, sticky=tk.NW)

        self.pgb_status.start(10)
        self.lbl_msg['text'] = 'please wait...'

        # ask the user to select their image
        fd_image_file = filedialog.askopenfilename()

        # testing if the file is a workable image
        if fd_image_file.endswith(('.tif', '.png', '.jpg', '.jpeg')):

            # split the file location into file name and path
            save_dir, file_name = os.path.split(fd_image_file)

            # read and load image from file
            self.orig_img = cv2.imread(os.path.join(save_dir, file_name), cv2.IMREAD_GRAYSCALE)

            # resize image to 300 pixel
            std_img = self.resize_img(300)

            # convert the Image object into a TkPhoto object
            img = Image.fromarray(std_img)
            img_tk = ImageTk.PhotoImage(image=img)

            # remove previous image and place the new one
            self.lbl_image.destroy()
            self.lbl_image = tk.Label(self, image=img_tk)
            self.lbl_image.image = img_tk
            self.lbl_image.grid(column=1, row=1, rowspan=12, padx=5, pady=2, sticky=tk.EW)

            # enable ChaosGT button and Select button
            self.btn_select_img['state'] = 'normal'
            self.btn_chaosgt['state'] = 'normal'
            self.pgb_status.stop()
            self.pgb_status.grid_forget()
            self.lbl_msg['text'] = 'Ready for chaos\nengineering!'
        else:
            self.error_msg = "Error: file needs tobe a\n.tif, .png, or .jpg"

            # disable ChaosGT button
            self.btn_select_img['state'] = 'normal'
            self.btn_chaosgt['state'] = 'disabled'
            self.pgb_status.stop()
            self.pgb_status.grid_forget()
            self.lbl_msg['text'] = self.error_msg

    def btn_chaosgt_clicked(self):
        pass

    def resize_img(self, size):
        w, h = self.orig_img.shape
        if (h > w):
            scalefactor = size / h
        else:
            scalefactor = size / w
        std_width = int(scalefactor * w)
        std_height = int(scalefactor * h)
        std_size = (std_height, std_width)
        std_img = cv2.resize(self.orig_img, std_size)
        return std_img


class ChaosGUI(tk.Tk):

    def __init__(self):
        super().__init__()

        # configure the root window
        self.title('Chaos Engineering in NP')
        self.geometry('640x360')
        self.resizable(0, 0)

        # layout for root window
        self.columnconfigure(0, weight=1)

        # load window components
        main_frame = MainFrame(self)
        main_frame.grid(column=0, row=0)
