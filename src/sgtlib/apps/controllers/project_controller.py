# SPDX-License-Identifier: GNU GPL v3
"""
Pyside6 (GUI components) controller class for managing the project.
"""

import os
import pickle
import logging
import requests
import numpy as np
from pathlib import Path
from packaging import version
from typing import TYPE_CHECKING, Optional
from PySide6.QtCore import Signal, Slot, QObject

if TYPE_CHECKING:
    # False at run time, only for a type-checker
    from _typeshed import SupportsWrite

from ..models.table_model import TableModel
from ... import __version__, __title__
from ...utils.sgt_utils import img_to_base64, verify_path
from ...imaging.image_processor import ALLOWED_IMG_EXTENSIONS, ALLOWED_GRAPH_FILE_EXTENSIONS


class ProjectController(QObject):

    projectOpenedSignal = Signal(str)

    def __init__(self, controller_obj, parent: QObject = None):
        super().__init__(parent)
        self._ctrl = controller_obj
        self._project_open = False

        # Project data
        self._project_data = {"name": "", "file_path": Path("")}
        self._software_update = "No updates available!"

        # Create Models
        self.imgThumbnailModel = TableModel([])
        
    def start_task(self):
        """Activate the wait flag and send a wait signal."""
        self._ctrl.wait_flag = True

    def stop_task(self):
        """Deactivate the wait flag and send a wait signal."""
        self._ctrl.wait_flag = False

    def save_project_data(self):
        """
        A handler function that handles saving project data.
        Returns: True if successful, False otherwise.

        """
        if not self._project_open:
            return False
        try:
            file_path = self._project_data["file_path"]
            with open(file_path, 'wb') as project_file:  # type: Optional[SupportsWrite[bytes]]
                pickle.dump(self._ctrl.sgt_objs, project_file)
            return True
        except Exception as err:
            logging.exception("Project Saving Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Save Error", "Unable to save project data. Close app and try again.")
            return False

    def get_thumbnail_list(self):
        """
        Get names and base64 data of images to be used in Project List thumbnails.
        """
        keys_list = list(self._ctrl.sgt_objs.keys())
        if len(keys_list) <= 0:
            return None, None
        item_data = []
        image_cache = {}
        for key in keys_list:
            item_data.append([key])  # Store the key
            sgt_obj = self._ctrl.sgt_objs[key]
            if sgt_obj.ntwk_p.selected_batch.is_graph_only:
                empty_cv = np.ones((256, 256), dtype=np.uint8) * 255
                img_cv = empty_cv if sgt_obj.ntwk_p.graph_obj.img_ntwk is None else sgt_obj.ntwk_p.graph_obj.img_ntwk
            else:
                img_cv = sgt_obj.ntwk_p.image_2d
            base64_data = img_to_base64(img_cv)
            image_cache[key] = base64_data  # Store base64 string
        return item_data, image_cache

    @Slot(result=bool)
    def is_project_open(self):
        return self._project_open

    @Slot(result=bool)
    def check_for_updates(self):
        """Check for updates and return True if an update is available, False otherwise."""
        github_url = "https://raw.githubusercontent.com/owuordickson/structural-gt/refs/heads/main/src/sgtlib/__init__.py"

        try:
            response = requests.get(github_url, timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            self._software_update = f"Error checking for updates: {e}"
            return False

        remote_version = None
        for line in response.text.splitlines():
            if line.strip().startswith("__install_version__"):
                try:
                    remote_version = line.split("=")[1].strip().strip("\"'")
                    break
                except IndexError:
                    self._software_update = "Could not connect to server!"
                    return False

        if not remote_version:
            self._software_update = "Could not find the new version!"
            return False

        new_version = version.parse(remote_version)
        current_version = version.parse(__version__)
        if new_version > current_version:
            # https://github.com/owuordickson/structural-gt/releases/tag/v3.3.5
            self._software_update = (
                "New version available!<br>"
                f"Download via this <a href='https://github.com/owuordickson/structural-gt/releases/tag/v{remote_version}'>link</a>"
            )
            return True
        else:
            self._software_update = "No updates available."
            return False

    @Slot(result=str)
    def get_sgt_title(self):
        return f"{__title__}"

    @Slot(result=str)
    def get_sgt_version(self):
        """"""
        # return f"{__title__} v{__version__}, Computing: {COMPUTING_DEVICE}"
        return f"v{__version__}"

    @Slot(result=str)
    def get_software_download_details(self):
        return self._software_update

    @Slot(result=str)
    def get_about_details(self):
        about_app = (
            "<html>"
            "<p>"
            "A software tool for performing Graph/Network Theory analysis on <br> "
            "microscopy images. This is a modified version of StructuralGT <br> "
            "initially proposed by D. Vecchio <br> "
            "DOI: <a href='https://pubs.acs.org/doi/10.1021/acsnano.1c04711'>10.1021/acsnano.1c04711</a>."
            "<br></p><p>"
            "<b>Main Contributors:</b>"
            "<table border='0.5' cellspacing='0' cellpadding='4'>"
            # "<tr><th>Name</th><th>Email</th></tr>"
            "<tr><td>Dickson Owuor</td><td>owuor@umich.edu</td></tr>"
            "<tr><td>Nicolas Kotov</td><td>kotov@umich.edu</td></tr>"
            "<tr><td>Alain Kadar</td><td>alaink@umich.edu</td></tr>"
            "<tr><td>Xiong Ye Xiao</td><td>xiongyex@usc.edu</td></tr>"
            "<tr><td>Kotov Lab</td><td></td></tr>"
            "<tr><td>COMPASS</td><td></td></tr>"
            "</table>"
            "<br></p><p>"
            "<b>Documentation:</b> <a href='https://structural-gt.readthedocs.io'>structural-gt.readthedocs.io</a>"
            "</p><p>"
            f"<b> Version: </b> {self.get_sgt_version()}"
            "</p><p>"
            "<b>License:</b> GPL GNU v3"
            "</p><p>"
            "<b>Icon Acknowledgements:</b>"
            "<ol>"
            "<li> <a href='https://www.iconfinder.com/'>IconFinder Library</a></li>"
            "<li> <a href='https://www.flaticon.com/'>Flaticon</a> </li>"
            "</ol>"
            "</p><p><br>"
            "Copyright (C) 2018-2025<br>The Regents of the University of Michigan."
            "</p>"
            "</html>")
        return about_app

    @Slot(str, result=str)
    def get_file_extensions(self, option):
        if option == "img":
            pattern_string = ' '.join(ALLOWED_IMG_EXTENSIONS)
            return f"Image files ({pattern_string})"
        if option == "graph":
            pattern_string = ' '.join(ALLOWED_GRAPH_FILE_EXTENSIONS)
            return f"Graph files ({pattern_string})"
        elif option == "proj":
            return "Project files (*.sgtproj)"
        else:
            return ""

    @Slot(result=str)
    def get_output_dir(self):
        sgt_obj = self._ctrl.get_selected_sgt_obj()
        if sgt_obj is None:
            return ""
        return f"{sgt_obj.ntwk_p.output_dir}"

    @Slot(str)
    def set_output_dir(self, folder_path):
        self._ctrl.update_output_dir(folder_path)
        self._ctrl.imageChangedSignal.emit()

    @Slot(int)
    def delete_selected_thumbnail(self, img_index):
        """Delete the selected image from the list."""
        self._ctrl.delete_sgt_object(img_index)

    @Slot(result=bool)
    def enable_prev_nav_btn(self):
        if (self._ctrl.selected_sgt_obj_index == 0) or self._ctrl.is_task_running():
            return False
        else:
            return True

    @Slot(result=bool)
    def enable_next_nav_btn(self):
        if (self._ctrl.selected_sgt_obj_index == (len(self._ctrl.sgt_objs) - 1)) or self._ctrl.is_task_running():
            return False
        else:
            return True

    @Slot(result=bool)
    def load_prev_image(self):
        """Load the previous image in the list into view."""
        if self._ctrl.selected_sgt_obj_index > 0:
            prev_img_idx = self._ctrl.selected_sgt_obj_index - 1
            self._ctrl.load_image(index=prev_img_idx)
            return True
        return False

    @Slot(result=bool)
    def load_next_image(self):
        """Load the next image in the list into view."""
        if self._ctrl.selected_sgt_obj_index < (len(self._ctrl.sgt_objs) - 1):
            next_img_idx = self._ctrl.selected_sgt_obj_index + 1
            self._ctrl.load_image(index=next_img_idx)
            return True
        return False

    @Slot(result=bool)
    def run_save_project(self):
        """"""
        if self._ctrl.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return False

        self.start_task()
        success_val = self.save_project_data()
        self.stop_task()
        return success_val

    @Slot(str, result=bool)
    def upload_graph_file(self, file_path):
        """Verify and validate the file path, use it to create a new SGT Object and load it into the view."""
        is_successful = self._ctrl.add_graph(file_path)
        if is_successful:
            self._ctrl.load_image(reload_thumbnails=True)
        return is_successful

    @Slot(str, result=bool)
    def upload_single_image(self, img_path):
        """Verify and validate the image path, use it to create an SGT object and load it in view."""
        is_successful = self._ctrl.add_single_image(img_path)
        if is_successful:
            self._ctrl.load_image(reload_thumbnails=True)
        return is_successful

    @Slot(str, result=bool)
    def upload_multiple_images(self, img_dir_path):
        """
        Verify and validate multiple image paths, use each to create an SGT object, then load the last one in view.
        """
        is_successful = self._ctrl.add_multiple_images(img_dir_path)
        if is_successful:
            self._ctrl.load_image(reload_thumbnails=True)
        return is_successful

    @Slot(str, str, result=bool)
    def create_sgt_project(self, proj_name, dir_path) -> bool:
        """Creates a '.sgtproj' inside the selected directory"""

        self._project_open = False
        success, result = verify_path(dir_path)
        if not success:
            logging.info(result, extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("File/Directory Error", result)
            return False

        dir_path = Path(result)
        proj_name = f"{proj_name}.sgtproj"
        proj_path = dir_path / proj_name

        try:
            if os.path.exists(proj_path):
                logging.info(f"Project '{proj_name}' already exists.", extra={'user': 'SGT Logs'})
                self._ctrl.showAlertSignal.emit("Project Error", f"Error: Project '{proj_name}' already exists.")
                return False

            # Create an empty project file (touch creates it if it doesn’t exist)
            proj_path.touch(exist_ok=False)

            # Update and notify QML
            self._project_data["name"] = proj_name
            self._project_data["file_path"] = proj_path
            self._project_open = True
            self.projectOpenedSignal.emit(proj_name)
            logging.info(f"File '{proj_name}' created successfully in '{dir_path}'.", extra={'user': 'SGT Logs'})
            return True
        except Exception as err:
            # self._project_open = False
            logging.exception("Create Project Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Create Project Error",
                                      "Failed to create SGT project. Close the app and try again.")
            return False

    @Slot(str, result=bool)
    def open_sgt_project(self, sgt_path):
        """Opens and loads the SGT project from the '.sgtproj' file"""
        if self._ctrl.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return False

        try:
            self.start_task()
            self._project_open = False
            # Verify the path
            success, result = verify_path(sgt_path)
            if success:
                sgt_path = result
            else:
                logging.info(result, extra={'user': 'SGT Logs'})
                self._ctrl.showAlertSignal.emit("File/Directory Error", result)
                self.stop_task()
                return False
            img_dir, proj_name = os.path.split(str(sgt_path))

            # Read and load project data and SGT objects
            with open(str(sgt_path), 'rb') as sgt_file:
                self._ctrl.sgt_objs = pickle.load(sgt_file)

            if self._ctrl.sgt_objs:
                key_list = list(self._ctrl.sgt_objs.keys())
                for key in key_list:
                    self._ctrl.sgt_objs[key].ntwk_p.output_dir = img_dir

            # Update and notify QML
            self._project_data["name"] = proj_name
            self._project_data["file_path"] = str(sgt_path)
            self.stop_task()
            self._project_open = True
            self.projectOpenedSignal.emit(proj_name)

            # Load Image to GUI - activates QML
            self._ctrl.load_image(reload_thumbnails=True)
            logging.info(f"File '{proj_name}' opened successfully in '{sgt_path}'.", extra={'user': 'SGT Logs'})
            return True
        except Exception as err:
            self.stop_task()
            logging.exception("Project Opening Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Open Project Error", "Unable to open .sgtproj file! Try again. If the "
                                                            "issue persists, the file may be corrupted or incompatible. "
                                                            "Consider restoring from a backup or contacting support for "
                                                            "assistance.")
            return False
