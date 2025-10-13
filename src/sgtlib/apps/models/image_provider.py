from PIL import Image, ImageQt  # Import ImageQt for conversion
from PySide6.QtGui import QPixmap
from PySide6.QtQuick import QQuickImageProvider


class ImageProvider(QQuickImageProvider):

    def __init__(self, img_controller):
        super().__init__(QQuickImageProvider.ImageType.Pixmap)
        self._pixmap = QPixmap()
        self._main_ctrl = img_controller
        self._main_ctrl.changeImageSignal.connect(self.handle_change_image)

    def handle_change_image(self):
        if len(self._main_ctrl.sgt_objs) > 0:
            img_cv = None
            sgt_obj = self._main_ctrl.get_selected_sgt_obj()
            ntwk_p = sgt_obj.ntwk_p
            sel_img_batch = ntwk_p.selected_batch
            self._main_ctrl.img_ctrl.showImageFilterControls.emit(True)
            if ntwk_p.selected_batch_view == "original":
                images = ntwk_p.image_3d
                if self._main_ctrl.img_ctrl.is_img_3d():
                    self._main_ctrl.img_ctrl.img3dGridModel.reset_data(images, sel_img_batch.selected_images_positions)
                else:
                    # 2D, Do not use if 3D
                    img_cv = images[0] if len(images) > 0 else None
            elif ntwk_p.selected_batch_view == "binary":
                # Apply filters
                ntwk_p.apply_img_filters(filter_type=2)
                bin_images = ntwk_p.binary_image_3d
                if self._main_ctrl.img_ctrl.is_img_3d():
                    self._main_ctrl.img_ctrl.img3dGridModel.reset_data(bin_images, sel_img_batch.selected_images_positions)
                else:
                    # 2D, Do not use if 3D
                    img_cv = bin_images[0] if len(bin_images) > 0 else None
            elif ntwk_p.selected_batch_view == "processed":
                # Apply filters
                ntwk_p.apply_img_filters(filter_type=1)
                mod_images = ntwk_p.processed_image_3d
                if self._main_ctrl.img_ctrl.is_img_3d():
                    self._main_ctrl.img_ctrl.img3dGridModel.reset_data(mod_images, sel_img_batch.selected_images_positions)
                else:
                    # 2D, Do not use if 3D
                    img_cv = mod_images[0] if len(mod_images) > 0 else None
            elif ntwk_p.selected_batch_view == "graph":
                # If any is None, start the task
                self._main_ctrl.img_ctrl.showImageFilterControls.emit(False)
                if ntwk_p.graph_obj.img_ntwk is None:
                    self._main_ctrl.graph_ctrl.run_extract_graph()
                    # Wait for the task to finish
                    return
                else:
                    net_images = [ntwk_p.graph_obj.img_ntwk]
                    self._main_ctrl.img_ctrl.img3dGridModel.reset_data(net_images, sel_img_batch.selected_images_positions)
                    img_cv = net_images[0]
            else:
                self._main_ctrl.img_ctrl.showImageFilterControls.emit(False)
                return

            if img_cv is not None:
                # Create Pixmap image
                img = Image.fromarray(img_cv)
                self._pixmap = ImageQt.toqpixmap(img)

            # Acknowledge the image load and send the signal to update QML
            self._main_ctrl.img_ctrl._img_loaded = True
            self._main_ctrl.imageChangedSignal.emit()
        else:
            self._main_ctrl.img_ctrl._img_loaded = False
        self._main_ctrl.img_ctrl._applying_changes = False

    def requestPixmap(self, img_id, requested_size, size):
        return self._pixmap
