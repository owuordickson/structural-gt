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
        if len(self.img_controller.analyze_objs) > 0:
            keys_list = list(self.img_controller.analyze_objs.keys())
            key_at_index = keys_list[self.img_controller.current_obj_index]
            a_obj = self.img_controller.analyze_objs[key_at_index]
            if option == "processed":
                img_cv = a_obj.g_obj.imp.img_mod
            elif option == "graph":
                img_cv = a_obj.g_obj.imp.img_net
            elif option == "crop":
                img_cv = a_obj.g_obj.imp.img
            elif option == "un-crop":
                a_obj.g_obj.imp.undo_cropping()
                img_cv = a_obj.g_obj.imp.img
            else:
                img_cv = a_obj.g_obj.imp.img
                # self.pixmap = self.pixmap_original
            img = Image.fromarray(img_cv)
            self.pixmap = ImageQt.toqpixmap(img)
            self.img_controller.img_loaded = True
            self.img_controller.imageChangedSignal.emit()  # signal to update QML image

    def requestPixmap(self, img_id, requested_size, size):
        # print(img_id)
        return self.pixmap

    def handle_change_image(self, cmd):
        # '0' - Original image
        # '1' - Cropped image
        # '2' - Processed image
        # '3' - Undo crop
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
            self.select_image("graph")
        elif cmd == 4:
            # Undo cropping (load original image)
            self.select_image("un-crop")
