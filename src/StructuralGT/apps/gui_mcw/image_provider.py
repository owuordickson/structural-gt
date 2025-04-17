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
            img = None
            sgt_obj = self.img_controller.get_selected_sgt_obj()
            ntwk_p = sgt_obj.ntwk_p
            sel_img_batch = ntwk_p.get_selected_batch()
            if option == "binary":
                ntwk_p.apply_img_filters(filter_type=2)
                bin_images = [obj.img_bin for obj in sel_img_batch.images]
                if self.img_controller.is_img_3d():
                    self.img_controller.img3dGridModel.reset_data(bin_images, sel_img_batch.selected_images)
                else:
                    # 2D, Do not use if 3D
                    img_cv = bin_images[0]
                    img = Image.fromarray(img_cv)
            elif option == "processed":
                ntwk_p.apply_img_filters(filter_type=1)
                mod_images = [obj.img_mod for obj in sel_img_batch.images]
                if self.img_controller.is_img_3d():
                    self.img_controller.img3dGridModel.reset_data(mod_images, sel_img_batch.selected_images)
                else:
                    # 2D, Do not use if 3D
                    img_cv = mod_images[0]
                    img = Image.fromarray(img_cv)
            elif option == "graph":

                # If any is None, start task
                if sel_img_batch.graph_obj.img_ntwk is None:
                    self.img_controller.run_extract_graph()
                    # Wait for the task to finish
                    return
                else:
                    net_images = [sel_img_batch.graph_obj.img_ntwk]
                    self.img_controller.img3dGridModel.reset_data(net_images, sel_img_batch.selected_images)
                    img = net_images[0]
                    self.img_controller.load_graph_simulation()
            else:
                # Original
                images = [obj.img_2d for obj in sel_img_batch.images]
                if self.img_controller.is_img_3d():
                    self.img_controller.img3dGridModel.reset_data(images, sel_img_batch.selected_images)
                else:
                    # 2D, Do not use if 3D
                    img_cv = images[0]
                    img = Image.fromarray(img_cv)

            if img is not None:
                # Create Pixmap image
                self.pixmap = ImageQt.toqpixmap(img)

            # Reset graph/image configs with selected values - reloads QML
            self.img_controller.update_graph_models(sgt_obj)

            # Save changes to the project data file
            if len(self.img_controller.sgt_objs.items()) <= 10:
                self.img_controller.save_project_data()

            # Acknowledge the image load and send signal to update QML
            self.img_controller.img_loaded = True
            self.img_controller.imageChangedSignal.emit()
        else:
            self.img_controller.img_loaded = False

    def requestPixmap(self, img_id, requested_size, size):
        return self.pixmap

    def handle_change_image(self, cmd):
        # '0' - Original image
        # '2' - Process image
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
