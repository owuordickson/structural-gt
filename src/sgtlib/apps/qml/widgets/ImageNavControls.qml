import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Effects
import QtQuick.Controls.Basic as Basic

Rectangle {
    id: imgNavControls
    height: 36
    Layout.fillHeight: false
    Layout.fillWidth: true
    Layout.alignment: Qt.AlignBottom | Qt.AlignHCenter
    Layout.margins: 5
    color: "#e5e5e5"
    radius: 5
    visible: imageController.display_image()

    // --- Exposed properties ---
    property int img_pos: 0
    property bool showPrev: true
    property bool showNext: true
    property bool showImgBatch: true
    property bool showImgPos: false

    property int selectedRole: Qt.UserRole + 20


    RowLayout {
        anchors.fill: parent
        anchors.verticalCenter: parent.verticalCenter

        Basic.Button {
            id: btnPrevious
            text: ""
            icon.source: "../assets/icons/back_icon.png"
            icon.width: 24
            icon.height: 24
            background: Rectangle {
                color: "transparent"
            }
            Layout.alignment: Qt.AlignLeft | Qt.AlignVCenter
            visible: imgNavControls.showPrev
            onClicked: projectController.load_prev_image()
        }

        Row {
            id: imgSelectionControls
            Layout.alignment: Qt.AlignCenter
            spacing: 4
            visible: imageController.image_batches_exist()

            ComboBox {
                id: cbBatchSelector
                Layout.minimumWidth: 75
                model: imgBatchModel
                implicitContentWidthPolicy: ComboBox.WidestTextWhenCompleted
                textRole: "text"
                valueRole: "value"
                ToolTip.text: "Change image batch"
                ToolTip.visible: cbBatchSelector.hovered
                visible: imgNavControls.showImgBatch
                onCurrentIndexChanged: imageController.select_img_batch(valueAt(currentIndex))
            }

            ComboBox {
                id: cbImageSelector
                Layout.minimumWidth: 75
                model: img3dGridModel
                implicitContentWidthPolicy: ComboBox.WidestTextWhenCompleted
                textRole: "text"
                valueRole: "id"
                ToolTip.text: "Select image"
                ToolTip.visible: cbImageSelector.hovered
                visible: imageController.is_img_3d() && imgNavControls.showImgPos
                currentIndex: imgNavControls.img_pos
                onCurrentIndexChanged: imageController.select_batch_image_index(valueAt(currentIndex))
            }
        }


        Basic.Button {
            id: btnNext
            text: ""
            icon.source: "../assets/icons/next_icon.png"
            icon.width: 24
            icon.height: 24
            background: Rectangle {
                color: "transparent"
            }
            Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
            visible: imgNavControls.showNext
            onClicked: projectController.load_next_image()
        }

    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            imgNavControls.visible = imageController.display_image();
            imgSelectionControls.visible = imageController.image_batches_exist();
            cbImageSelector.visible = imageController.is_img_3d() && imgNavControls.showImgPos;

            btnPrevious.enabled = projectController.enable_prev_nav_btn();
            btnNext.enabled = projectController.enable_next_nav_btn();

            if (imageController.image_batches_exist() && imgNavControls.showImgBatch) {
                cbBatchSelector.currentIndex = imageController.get_selected_img_batch();
            }

            if (imageController.is_img_3d() && imgNavControls.showImgPos) {
                imgNavControls.img_pos = imageController.get_selected_batch_image_index();
            } else {
                imgNavControls.img_pos = 0;
            }

            if (imgNavControls.showImgPos) {
                // Completely hide NavControls if it is not a 3D image
                imgNavControls.visible = imageController.is_img_3d();
            }

        }

        function onUpdateProgressSignal(val, msg) {
            btnNext.enabled = projectController.enable_next_nav_btn();
        }

        function onTaskTerminatedSignal(success_val, msg_data) {
            btnNext.enabled = projectController.enable_next_nav_btn();
        }

    }

}