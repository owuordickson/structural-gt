from PIL import Image, ImageQt  # Import ImageQt for conversion
from PySide6.QtGui import QPixmap
from PySide6.QtQuick import QQuickImageProvider


class ImageProvider(QQuickImageProvider):

    def __init__(self, img_controller):
        super().__init__(QQuickImageProvider.ImageType.Pixmap)
        self.pixmap = QPixmap()
        self.img_controller = img_controller
        self.img_controller.changeImageSignal.connect(self.handle_change_image)

    def handle_change_image(self):
        if len(self.img_controller.sgt_objs) > 0:
            img_cv = None
            sgt_obj = self.img_controller.get_selected_sgt_obj()
            ntwk_p = sgt_obj.ntwk_p
            sel_img_batch = ntwk_p.get_selected_batch()
            if sel_img_batch.current_view == "binary":
                # Apply filters
                ntwk_p.apply_img_filters(filter_type=2)
                # Calculate image histogram in different thread
                self.img_controller.compute_img_histogram()
                bin_images = ntwk_p.binary_image_3d
                if self.img_controller.is_img_3d():
                    self.img_controller.img3dGridModel.reset_data(bin_images, sel_img_batch.selected_images_idx)
                else:
                    # 2D, Do not use if 3D
                    img_cv = bin_images[0]
            elif sel_img_batch.current_view  == "processed":
                # Apply filters
                ntwk_p.apply_img_filters(filter_type=1)
                # Calculate image histogram in different thread
                self.img_controller.compute_img_histogram()
                mod_images = ntwk_p.modified_image_3d
                if self.img_controller.is_img_3d():
                    self.img_controller.img3dGridModel.reset_data(mod_images, sel_img_batch.selected_images_idx)
                else:
                    # 2D, Do not use if 3D
                    img_cv = mod_images[0]
            elif sel_img_batch.current_view  == "graph":
                # If any is None, start the task
                self.img_controller.showImageHistogramSignal.emit(False)
                if sel_img_batch.graph_obj.img_ntwk is None:
                    self.img_controller.run_extract_graph()
                    # Wait for the task to finish
                    return
                else:
                    net_images = [sel_img_batch.graph_obj.img_ntwk]
                    self.img_controller.img3dGridModel.reset_data(net_images, sel_img_batch.selected_images_idx)
                    img_cv = net_images[0]
            else:
                # Original
                # Calculate image histogram in different thread
                self.img_controller.compute_img_histogram()
                images = ntwk_p.image_3d
                if self.img_controller.is_img_3d():
                    self.img_controller.img3dGridModel.reset_data(images, sel_img_batch.selected_images_idx)
                else:
                    # 2D, Do not use if 3D
                    img_cv = images[0]

            if img_cv is not None:
                # Create Pixmap image
                img = Image.fromarray(img_cv)
                self.pixmap = ImageQt.toqpixmap(img)

            # Reset graph/image configs with selected values - reloads QML
            self.img_controller.update_graph_models(sgt_obj)

            # Acknowledge the image load and send the signal to update QML
            self.img_controller.img_loaded = True
            self.img_controller.imageChangedSignal.emit()
        else:
            self.img_controller.img_loaded = False

    def requestPixmap(self, img_id, requested_size, size):
        return self.pixmap
