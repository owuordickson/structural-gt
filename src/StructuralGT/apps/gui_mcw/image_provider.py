from PIL import Image, ImageQt  # Import ImageQt for conversion
from PySide6.QtGui import QPixmap
from PySide6.QtQuick import QQuickImageProvider


class ImageProvider(QQuickImageProvider):

    def __init__(self, img_controller):
        super().__init__(QQuickImageProvider.Pixmap)
        self.pixmap = QPixmap()
        self.img_controller = img_controller
        self.img_controller.changeImageSignal.connect(self.handle_change_image)

    def select_image(self, option: str=""):
        if len(self.img_controller.sgt_objs) > 0:
            sgt_obj = self.img_controller.get_current_obj()
            im_obj = sgt_obj.g_obj.imp
            if option == "crop":
                img_cv = im_obj.img
                img = Image.fromarray(img_cv)
            elif option == "un-crop":
                im_obj.undo_cropping()
                img_cv = im_obj.img
                img = Image.fromarray(img_cv)
            elif option == "binary":
                im_obj.img_mod = im_obj.process_img(im_obj.img.copy())
                im_obj.img_bin = im_obj.binarize_img(im_obj.img_mod.copy())
                img_cv = im_obj.img_bin
                img = Image.fromarray(img_cv)
            elif option == "processed":
                im_obj.img_mod = im_obj.process_img(im_obj.img.copy())
                img_cv = im_obj.img_mod
                img = Image.fromarray(img_cv)
            elif option == "graph":
                if im_obj.img_net is None:
                    self.img_controller.run_extract_graph()
                    # Wait for task to finish
                    return
                else:
                    img = im_obj.img_net
            else:
                img_cv = im_obj.img
                img = Image.fromarray(img_cv)
            self.pixmap = ImageQt.toqpixmap(img)
            # Reset graph/image configs with selected values - reloads QML
            self.img_controller.update_graph_models(sgt_obj)
            # Save changes to project data file
            if len(self.img_controller.sgt_objs.items()) <= 10:
                self.img_controller.save_project_data()

            # Acknowledge image load and send signal to update QML
            self.img_controller.img_loaded = True
            self.img_controller.imageChangedSignal.emit()

    def requestPixmap(self, img_id, requested_size, size):
        return self.pixmap

    def handle_change_image(self, cmd):
        # '0' - Original image
        # '1' - Cropping image
        # '2' - Processing image
        # '3' - Binarize image
        # '4' - Extract graph
        # '5' - Undo crop
        if cmd == 0:
            # Original QPixmap image for first time
            self.select_image("")
        elif cmd == 1:
            # cropped QPixmap image
            self.select_image("crop")
        elif cmd == 2:
            # processed QPixmap image
            self.select_image("processed")
        elif cmd == 3:
            self.select_image("binary")
        elif cmd == 4:
            self.select_image("graph")
        elif cmd == 5:
            # Undo cropping (load original image)
            self.select_image("un-crop")
