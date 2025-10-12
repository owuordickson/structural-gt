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
    //opacity: 0.5
    radius: 5
    visible: imageController.display_image()

    // Expose to outside QMLs
    property alias cbImageSelector: cbImageSelector


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
                onCurrentIndexChanged: imageController.select_img_batch(valueAt(currentIndex))
            }

            Rectangle {
                id: vertNavLine
                width: 1
                height: 14
                color: "#808080"
                anchors.verticalCenter: parent.verticalCenter
                visible: imageController.is_img_3d()
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
                visible: imageController.is_img_3d()
                currentIndex: imageController.get_selected_batch_image_index()
                onCurrentIndexChanged: {
                    let index = img3dGridModel.index(model.index, 0);
                    let selectedVal = 1;
                    //img3dGridModel.setData(index, selectedVal, selectedRole);
                    //imageController.select_batch_image_index(model.id);
                }
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
            onClicked: projectController.load_next_image()
        }

    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            imgNavControls.visible = imageController.display_image();
            cbBatchSelector.visible = imageController.image_batches_exist();
            imgSelectionControls.visible = imageController.image_batches_exist();
            vertNavLine.visible = imageController.is_img_3d();
            cbImageSelector.visible = imageController.is_img_3d();

            btnPrevious.enabled = projectController.enable_prev_nav_btn();
            btnNext.enabled = projectController.enable_next_nav_btn();

            if (imageController.image_batches_exist()) {
                cbBatchSelector.currentIndex = imageController.get_selected_img_batch();
            }

            /*let img_pos = 0;
            if (imageController.is_img_3d()) {
                cbImageSelector.currentIndex = imageController.get_selected_batch_image_index();
                img_pos = cbImageSelector.currentIndex;
            }*/



        }

        function onUpdateProgressSignal(val, msg) {
            btnNext.enabled = projectController.enable_next_nav_btn();
        }

        function onTaskTerminatedSignal(success_val, msg_data) {
            btnNext.enabled = projectController.enable_next_nav_btn();
        }

    }

}