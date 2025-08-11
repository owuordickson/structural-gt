from PIL import Image, ImageQt
from PyQt6.QtGui import QPixmap
from PyQt6.QtQuick import QQuickImageProvider


class ImageProvider(QQuickImageProvider):

    def __init__(self, img_controller):
        super().__init__(QQuickImageProvider.ImageType.Pixmap)
        self.pixmap = QPixmap()
        self.img_controller = img_controller
        self.img_controller.changeImageSignal.connect(self.change_image)

    def change_image(self):
        # Create Pixmap image
        img = Image.fromarray(self.img_controller.img_cv)
        self.pixmap = ImageQt.toqpixmap(img)
        self.img_controller.imageChangedSignal.emit(True)

    def requestPixmap(self, img_id, requested_size):
        return self.pixmap, self.pixmap.size()
