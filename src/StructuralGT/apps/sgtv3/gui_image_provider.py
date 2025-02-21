from PySide6.QtGui import QPixmap
from PySide6.QtQuick import QQuickImageProvider


class ImageProvider(QQuickImageProvider):

    def __init__(self, img_controller):
        super().__init__(QQuickImageProvider.Pixmap)
        self.pixmap = QPixmap()
        self.pixmap_original = QPixmap()
        self.pixmap_cropped = QPixmap()
        self.pixmap_processed = QPixmap()
        self.img_controller = img_controller
        self.img_controller.changeImageSignal.connect(self.handle_change_image)

    def select_image(self, option: str=""):
        if option == "crop":
            self.pixmap = self.pixmap_cropped
        elif option == "processed":
            self.pixmap = self.pixmap_processed
        else:
            self.pixmap = self.pixmap_original
        self.img_controller.img_loaded = True
        self.img_controller.imageChangedSignal.emit()  # signal to update QML image

    def requestPixmap(self, img_id, requested_size, size):
        # print(img_id)
        return self.pixmap

    def handle_change_image(self, cmd, img_pixmap):
        # '0' - Original image
        # '1' - Cropped image
        # '2' - Processed image
        # '3' - Undo crop,
        if cmd == 0:
            # Original QPixmap image for first time
            self.pixmap_original = img_pixmap
            self.select_image("")
        elif cmd == 1:
            # cropped QPixmap image
            self.pixmap_cropped = img_pixmap
            self.select_image("crop")
        elif cmd == 2:
            # processed QPixmap image
            self.pixmap_processed = img_pixmap
            self.select_image("processed")
        elif cmd == 3:
            # Undo (load original image)
            self.select_image("")
