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
            if option == "processed":
                img_cv = sgt_obj.g_obj.imp.img_mod
            elif option == "graph":
                img_cv = sgt_obj.g_obj.imp.img_net
            elif option == "crop":
                img_cv = sgt_obj.g_obj.imp.img
            elif option == "un-crop":
                sgt_obj.g_obj.imp.undo_cropping()
                img_cv = sgt_obj.g_obj.imp.img
            else:
                img_cv = sgt_obj.g_obj.imp.img
            img = Image.fromarray(img_cv)
            self.pixmap = ImageQt.toqpixmap(img)
            self.img_controller.load_img_configs(sgt_obj)
            self.img_controller.img_loaded = True
            self.img_controller.imageChangedSignal.emit()  # signal to update QML image

    def requestPixmap(self, img_id, requested_size, size):
        return self.pixmap

    def handle_change_image(self, cmd):
        # '0' - Original image
        # '1' - Cropped image
        # '2' - Processed image
        # '3' - Binary image
        # '4' - Extracted graph
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
