"""
**ChaosGT**

A python package for chaos engineering in nano-structures.

Copyright (C) 2024, The University of Michigan.

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Development Lead: Dickson Owuor
Contributors: Nicholas A. Kotov
Contact email: owuordickson@gmail.com
"""

__author__ = "Dickson Owuor"
__credits__ = "Chemical Engineering Department, University of Michigan"


import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


class MainFrame(ttk.Frame):

    def __init__(self, container):
        super().__init__(container)

        # variables to store path and dir
        self.img_path = ''
        self.save_dir = ''
        self.error_msg = ''

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

        # (button) Select Image
        self.btn_select_img = tk.Button(self, text='Select Image', width=15, height=2, compound='c')
        self.btn_select_img['fg'] = 'dark green'
        self.btn_select_img['command'] = self.btn_select_img_clicked
        self.btn_select_img.grid(column=0, row=1, padx=5, pady=2, sticky=tk.EW)

        # (button) ChaosGT Image
        self.btn_chaosgt = tk.Button(self, text='Proceed to Chaos GT', width=15, height=2, compound='c')
        self.btn_chaosgt['fg'] = 'dark green'
        self.btn_chaosgt['command'] = self.btn_chaosgt_clicked
        self.btn_chaosgt['state'] = 'disabled'
        self.btn_chaosgt.grid(column=0, row=2, padx=5, pady=2, sticky=tk.NW)

        # Image Placeholder
        self.cvs_image = tk.Canvas(self, width=300, height=300, bg='white')
        self.cvs_image.grid(column=1, row=1, rowspan=12, padx=5, pady=2, sticky=tk.EW)

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

        # ask the user to select their image
        fd_image = filedialog.askopenfilename()

        # testing if the file is a workable image
        if fd_image.endswith(('.tif', '.png', '.jpg', '.jpeg')):

            # removing the prvious canvas image
            # cnv_list = self.slaves
            # for cnv in cnv_list:
            #    if isinstance(cnv, tk.Canvas):
            #        cnv.destroy()
            
            # split the file location into file name and path
            self.save_dir, self.img_path = os.path.split(fd_image)

            # enable ChaosGT button and Select button
            self.btn_select_img['state'] = 'normal'
            self.btn_chaosgt['state'] = 'normal'
        else:
            self.error_msg = "Error: file needs to be a .tif, .png, or .jpg"

            # disable ChaosGT button
            self.btn_select_img['state'] = 'normal'
            self.btn_chaosgt['state'] = 'disabled'


    def btn_chaosgt_clicked(self):
        pass



class ChaosApp(tk.Tk):

    def __init__(self):
        super().__init__()

        # configure the root window
        self.title('Chaos Engineering in NP')
        self.geometry('640x360')
        self.resizable(0, 0)

        # layout for root window
        self.columnconfigure(0, weight=1)

        main_frame = MainFrame(self)
        main_frame.grid(column=0, row=0)


if __name__ == "__main__":

    app = ChaosApp()
    app.mainloop()
