# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Tk-inter window implementations
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from .chaosgt import GraphStruct


class LaunchWindow(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)

        # about information
        self.about_info = "Chaos-GT is an application for implementing chaos engineering in nano-structures. " \
                          "The application was designed by Dickson Owuor and Nicholas Kotov." \
                          "\n\n\n" \
                          "Copyright (C) 2024, University of Michigan." \
                          "\n\n"

        # variables and objects
        self.parent = parent
        self.chaos_obj = None
        self.error_msg = ''

        # set up grid layout
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=3)

        # initialize widgets
        self.__create_widgets()

    # noinspection PyTypeChecker
    def __create_widgets(self):

        # (button) About
        self.btn_about = ttk.Button(self, text='About')
        self.btn_about['command'] = self.__btn_about_clicked
        self.btn_about.grid(column=1, row=0, padx=5, pady=5, sticky=tk.E)

        # (button) Select Image File
        self.btn_select_img = tk.Button(self, text='Select Image', width=15, height=2, compound='c')
        self.btn_select_img['fg'] = 'dark green'
        self.btn_select_img['command'] = self.__btn_select_img_clicked
        self.btn_select_img.grid(column=0, row=1, padx=5, pady=2, sticky=tk.EW)

        # (button) ChaosGT Algorithm
        self.btn_proceed = tk.Button(self, text='Proceed to Chaos GT', width=15, height=2, compound='c')
        self.btn_proceed['fg'] = 'dark green'
        self.btn_proceed['command'] = self.__btn_proceed_clicked
        self.btn_proceed['state'] = 'disabled'
        self.btn_proceed.grid(column=0, row=2, padx=5, pady=2, sticky=tk.NW)

        # (label) Show status/error
        self.pgb_status = ttk.Progressbar(self, mode='indeterminate')
        # self.pgb_status.grid(column=0, row=3, padx=5, pady=5, sticky=tk.SW)
        self.pgb_status.grid_forget()  # hide temporarily

        self.lbl_msg = tk.Label(self, text='', fg='gray')
        # self.lbl_msg.grid(column=0, row=4, padx=5, pady=2, sticky=tk.NW)
        self.lbl_msg.grid_forget()  # hide temporarily

        # Image Placeholder
        pixel = tk.PhotoImage(width=1, height=1)
        self.lbl_image = tk.Label(self, image=pixel, width=300, height=300, bg='white')
        self.lbl_image.grid(column=1, row=1, rowspan=12, padx=5, pady=2, sticky=tk.EW)

    def __btn_about_clicked(self):
        messagebox.showinfo(
            title='About Chaos GT',
            message=self.about_info
        )

    def __btn_select_img_clicked(self):

        # disable btn_select until image is loaded
        self.btn_select_img['state'] = 'disabled'
        self.btn_proceed['state'] = 'disabled'
        self.pgb_status.grid(column=0, row=4, padx=5, pady=1, sticky=tk.SW)
        self.lbl_msg.grid(column=0, row=5, padx=5, pady=1, sticky=tk.NW)

        self.pgb_status.start(10)
        self.lbl_msg['text'] = 'please wait...'

        # ask the user to select their image
        fd_image_file = filedialog.askopenfilename()

        # testing if the file is a workable image
        if fd_image_file.endswith(('.tiff', '.png', '.jpg', '.jpeg')):

            # split the file location into file name and path
            save_dir, file_name = os.path.split(fd_image_file)

            # load image from file and resize it to 300 pixel
            self.chaos_obj = GraphStruct(os.path.join(save_dir, file_name))
            std_img = self.chaos_obj.resize_img(300)
            self.chaos_obj.compute_fractal_dimension()

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
            self.btn_proceed['state'] = 'normal'
            self.pgb_status.stop()
            self.pgb_status.grid_forget()
            self.lbl_msg['text'] = 'Ready for chaos\nengineering!'
        else:
            self.error_msg = "Error: file needs tobe a\n.tif, .png, or .jpg"

            # disable ChaosGT button
            self.btn_select_img['state'] = 'normal'
            self.btn_proceed['state'] = 'disabled'
            self.pgb_status.stop()
            self.pgb_status.grid_forget()
            self.lbl_msg['text'] = self.error_msg

    def __btn_proceed_clicked(self):

        # re-configure the root window
        self.parent.title('Nano Structure Analysis')
        self.parent.geometry('800x600')

        # initialize and load window for analysis
        analytic_window = AnalysisWindow(self.parent, self.chaos_obj)
        analytic_window.grid(column=0, row=0)


class AnalysisWindow(ttk.Frame):

    def __init__(self, parent, c_obj):
        super().__init__(parent)

        # variables
        self.parent = parent
        self.error_msg = ''
        self.chaos_obj = c_obj

        # set up grid layout
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=1)

        # initialize widgets
        self.__create_widgets()

    def __create_widgets(self):

        # convert the Image object into a TkPhoto object
        std_img = self.chaos_obj.resize_img(512)
        img = Image.fromarray(std_img)
        img_tk = ImageTk.PhotoImage(image=img)

        # Titles
        self.lbl_title_res = tk.Label(self, text="Visualize Results", font="Bold")
        self.lbl_title_res.grid(column=0, row=0, padx=2, pady=2, sticky=tk.W)
        self.lbl_title_set = tk.Label(self, text="Modify Settings", font="Bold")
        self.lbl_title_set.grid(column=1, row=0, padx=50, pady=2, sticky=tk.W)

        # Image
        self.lbl_image = tk.Label(self, image=img_tk)
        self.lbl_image.image = img_tk
        self.lbl_image.grid(column=0, row=1, rowspan=4, padx=2, pady=2, sticky=tk.NW)

        self.lbl_text = tk.Label(self, text="start analysis")
        self.lbl_text.grid(column=1, row=1, padx=2, pady=2, sticky=tk.EW)


class ChaosGUI(tk.Tk):

    def __init__(self):
        super().__init__()

        # configure the root window
        self.title('Chaos Engineering in NP')
        self.geometry('640x360')
        # noinspection PyTypeChecker
        self.resizable(0, 0)

        # layout for root window
        self.columnconfigure(0, weight=1)

        # load window components
        splash_frame = LaunchWindow(self)
        splash_frame.grid(column=0, row=0)
