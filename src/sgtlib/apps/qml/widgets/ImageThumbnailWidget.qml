import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Basic as Basic


ColumnLayout {
    id: imgThumbsLayout
    Layout.preferredHeight: 512
    Layout.preferredWidth: parent.width
    Layout.leftMargin: 5
    Layout.rightMargin: 5
    Layout.bottomMargin: 10
    spacing: 5

    property int numRows: 10
    property int tblRowHeight: 50


    Text {
        text: "Loaded Images"
        font.pixelSize: 12
        font.bold: true
        Layout.alignment: Qt.AlignHCenter | Qt.AlignTop
        visible: true
    }

    Label {
        id: lblNoImages
        Layout.alignment: Qt.AlignHCenter | Qt.AlignTop
        text: "No images to show!\nPlease add image/folder."
        color: "#808080"
        visible: imgThumbnailModel.rowCount() <= 0
    }


    TableView {
        id: tblImgThumbs
        height: imgThumbsLayout.height - tblRowHeight
        width: parent.width
        rowSpacing: 2
        model: imgThumbnailModel
        visible: false

        delegate: Rectangle {
            implicitWidth: tblImgThumbs.width
            implicitHeight: tblRowHeight
            //color: row % 2 === 0 ? "#f5f5f5" : "#ffffff" // Alternating colors
            color: model.selected ? "#d0d0d0" : "transparent"

            MouseArea {
                anchors.fill: parent
                onClicked: mainController.load_image(row)
            }

            RowLayout {
                anchors.fill: parent
                spacing: 6

                Rectangle {
                    width: tblRowHeight
                    height: tblRowHeight
                    radius: 4
                    color: "transparent"
                    border.width: 1
                    border.color: "black"

                    Image {
                        id: imgThumbnail
                        anchors.fill: parent
                        source: "data:image/png;base64," + model.thumbnail  // Base64 encoded image
                    }

                }


                Text {
                    id: txtImgItem
                    Layout.fillWidth: true                 // ✅ take remaining space, not more
                    Layout.maximumWidth: parent.width - 100 // ✅ leave space for delete button
                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                    text: model.text
                    wrapMode: Text.NoWrap
                    elide: Text.ElideRight
                    maximumLineCount: 1        // ensures single-line behavior
                    font.pixelSize: 14
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignLeft
                    clip: true
                }

                Basic.Button {
                    id: btnDelete
                    Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                    text: ""
                    icon.source: "../assets/icons/delete_icon.png"
                    icon.width: 21
                    icon.height: 21
                    icon.color: "transparent"   // important for PNGs
                    background: Rectangle {
                        color: "transparent"
                    }
                    ToolTip.text: "Delete image."
                    ToolTip.visible: btnDelete.hovered
                    visible: model.selected
                    onClicked: mainController.delete_selected_thumbnail(row)
                }

            }

        }

    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            lblNoImages.visible = imgThumbnailModel.rowCount() <= 0
            tblImgThumbs.visible = imgThumbnailModel.rowCount() > 0
            tblImgThumbs.enabled = !mainController.is_task_running();

        }

        function onProjectOpenedSignal(name) {
            lblNoImages.text = "No images to show!\nPlease import image(s).";
            tblImgThumbs.visible = false;
        }

        function onUpdateProgressSignal(val, msg) {
            tblImgThumbs.enabled = !mainController.is_task_running();
        }

        function onTaskTerminatedSignal(success_val, msg_data) {
            tblImgThumbs.enabled = !mainController.is_task_running();
        }

    }

}
