import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Effects
import QtQuick.Controls.Basic as Basic

Rectangle {
    id: imgNavControls
    height: 32
    Layout.fillHeight: false
    Layout.fillWidth: true
    Layout.alignment: Qt.AlignBottom
    color: "transparent"
    visible: imageController.display_image()


    RowLayout {
        anchors.fill: parent

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

        Label {
            id: lblNavInfo
            text: ""
            color: "#808080"
            Layout.alignment: Qt.AlignCenter
        }


        ComboBox {
            id: cbBatchSelector
            visible: imageController.image_batches_exist()
            Layout.minimumWidth: 75
            model: imgBatchModel
            implicitContentWidthPolicy: ComboBox.WidestTextWhenCompleted
            textRole: "text"
            valueRole: "value"
            ToolTip.text: "Change image batch"
            ToolTip.visible: cbBatchSelector.hovered
            onCurrentIndexChanged: imageController.select_img_batch(valueAt(currentIndex))
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
            imgNavControls.visible = true//imageController.display_image();
            cbBatchSelector.visible = true//imageController.image_batches_exist();

            btnPrevious.enabled = projectController.enable_prev_nav_btn();
            btnNext.enabled = projectController.enable_next_nav_btn();
            lblNavInfo.text = projectController.get_img_nav_location();

            cbBatchSelector.currentIndex = imageController.get_selected_img_batch();
        }

        function onUpdateProgressSignal(val, msg) {
            if (val === 101) {
                lblNavInfo.text = msg;
            }
            btnNext.enabled = projectController.enable_next_nav_btn();
            lblNavInfo.text = projectController.get_img_nav_location();
        }

        function onTaskTerminatedSignal(success_val, msg_data) {
            lblNavInfo.text = projectController.get_img_nav_location();
            btnNext.enabled = projectController.enable_next_nav_btn();
            lblNavInfo.text = projectController.get_img_nav_location();
        }

    }

}