from PySide6.QtGui import QPixmap
from PySide6.QtQuick import QQuickImageProvider


class ImageProvider(QQuickImageProvider):

    def __init__(self, img_controller):
        super().__init__(QQuickImageProvider.Pixmap)
        # self.images = {}  # Store images with their IDs
        self._image_path = ""
        self.pixmap = QPixmap()
        self.pixmap_original = QPixmap()
        self.pixmap_cropped = QPixmap()
        self.img_controller = img_controller
        self.img_controller.imageChangedSignal.connect(self.handle_change_image)

    def set_image(self, image_path: str, option: str =""):
        self._image_path = image_path
        if option == "crop":
            self.pixmap_cropped.load(image_path)
        else:
            self.pixmap_original.load(image_path)
        self.img_controller.img_loaded = False
        # print(image_path)

    def select_image(self, option: str=""):
        if option == "crop":
            self.pixmap = self.pixmap_cropped
        else:
            self.pixmap = self.pixmap_original
        self.img_controller.img_loaded = True

    def requestPixmap(self, img_id, requested_size, size):
        # print(img_id)
        return self.pixmap

    def handle_change_image(self, src, img_path):
        # '0' - Original image
        # '1' - Cropped image
        # '2' - Processed image
        # '3'-Undo crop,
        # ignore '-1' - will make function recursive
        if src == 0:
            self.set_image(img_path, "")
            self.select_image("")
            self.img_controller.imageChangedSignal.emit(-1, img_path)  # signal to update QML image
        elif src == 1:
            self.set_image(img_path, "crop")
            self.select_image("crop")
            self.img_controller.imageChangedSignal.emit(-1, img_path)  # signal to update QML image
        elif src == 2:
            self.set_image(img_path, "process")
            self.select_image("process")
            self.img_controller.imageChangedSignal.emit(-1, img_path) # signal to update QML image
        elif src == 3:
            self.select_image("")
            self.img_controller.imageChangedSignal.emit(-1, img_path)  # signal to update QML image
