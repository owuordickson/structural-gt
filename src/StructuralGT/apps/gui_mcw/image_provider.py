from PIL import Image, ImageQt  # Import ImageQt for conversion
from PySide6.QtGui import QPixmap
from PySide6.QtQuick import QQuickImageProvider


class ImageProvider(QQuickImageProvider):

    def __init__(self, img_controller):
        super().__init__(QQuickImageProvider.ImageType.Pixmap)
        self.pixmap = QPixmap()
        self.img_controller = img_controller
        self.img_controller.changeImageSignal.connect(self.handle_change_image)

    def select_image(self, option: str=""):
        if len(self.img_controller.sgt_objs) > 0:
            sgt_obj = self.img_controller.get_current_obj()
            im_obj = sgt_obj.imp
            if option == "binary":
                im_obj.apply_img_filters(filter_type=2)
                bin_images = [obj.img_bin for obj in im_obj.images]
                self.img_controller.img3dGridModel.reset_data(bin_images)
                img_cv = bin_images[0]
                img = Image.fromarray(img_cv)
            elif option == "processed":
                im_obj.apply_img_filters(filter_type=1)
                mod_images = [obj.img_mod for obj in im_obj.images]
                self.img_controller.img3dGridModel.reset_data(mod_images)
                img_cv = mod_images[0]
                img = Image.fromarray(img_cv)
            elif option == "graph":
                for img in im_obj.images:
                    if img.graph_obj.img_net is None:
                        self.img_controller.run_extract_graph()
                        # Wait for task to finish
                        return
                else:
                    net_images = [obj.graph_obj.img_net for obj in im_obj.images]
                    self.img_controller.img3dGridModel.reset_data(net_images)
                    img = net_images[0]
            else:
                # Original
                images = [obj.img_2d for obj in im_obj.images]
                self.img_controller.img3dGridModel.reset_data(images)
                img_cv = images[0]
                img = Image.fromarray(img_cv)

            # Create Pixmap image
            self.pixmap = ImageQt.toqpixmap(img)

            # Reset graph/image configs with selected values - reloads QML
            self.img_controller.update_graph_models(sgt_obj)

            # Save changes to project data file
            if len(self.img_controller.sgt_objs.items()) <= 10:
                self.img_controller.save_project_data()

            # Acknowledge image load and send signal to update QML
            self.img_controller.img_loaded = True
            self.img_controller.imageChangedSignal.emit()
        else:
            self.img_controller.img_loaded = False

    def requestPixmap(self, img_id, requested_size, size):
        return self.pixmap

    def handle_change_image(self, cmd):
        # '0' - Original image
        # '2' - Processing image
        # '3' - Binarize image
        # '4' - Extract graph
        if cmd == 0:
            # Original QPixmap image for first time
            self.select_image("")
        elif cmd == 2:
            # processed QPixmap image
            self.select_image("processed")
        elif cmd == 3:
            self.select_image("binary")
        elif cmd == 4:
            self.select_image("graph")
