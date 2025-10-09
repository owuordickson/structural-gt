import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Basic as Basic


ColumnLayout {
    id: imgThumbsLayout
    Layout.preferredHeight: 512
    Layout.preferredWidth: parent.width
    Layout.alignment: Qt.AlignTop
    Layout.leftMargin: 5
    Layout.rightMargin: 5
    Layout.bottomMargin: 10
    spacing: 5

    property int tblRowHeight: 50


    Text {
        text: "Loaded Images"
        font.pixelSize: 12
        font.bold: true
        Layout.alignment: Qt.AlignHCenter | Qt.AlignTop
        Layout.bottomMargin: 5
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
        width: 290
        //height: 450
        Layout.fillHeight: true
        Layout.preferredHeight: 450
        clip: true
        rowSpacing: 2
        columnSpacing: 4
        visible: imgThumbnailModel.rowCount() > 0
        model: imgThumbnailModel
        selectionBehavior: TableView.SelectRows
        boundsBehavior: Flickable.StopAtBounds

        delegate: Rectangle {
            implicitWidth: tblImgThumbs.width
            implicitHeight: tblRowHeight + 10
            //color: model.selected ? "#d0d0d0" : (row % 2 === 0 ? "#fafafa" : "white")
            color: model.selected ? "#d0d0d0" : "transparent"

            MouseArea {
                anchors.fill: parent
                onClicked: mainController.load_image(row)
            }

            RowLayout {
                anchors.fill: parent
                anchors.margins: 4
                spacing: 6

                Rectangle {
                    width: tblRowHeight
                    height: tblRowHeight
                    radius: 4
                    color: "transparent"
                    border.width: 1
                    border.color: "#404040"

                    Image {
                        anchors.fill: parent
                        source: "data:image/png;base64," + model.thumbnail
                        fillMode: Image.PreserveAspectCrop
                        asynchronous: true  // ✅ prevents UI freeze when loading images
                        cache: false        // ✅ prevents stale images
                    }
                }

                Text {
                    Layout.fillWidth: true
                    Layout.preferredWidth: tblImgThumbs.width - tblRowHeight - 40
                    text: model.text
                    wrapMode: Text.NoWrap
                    elide: Text.ElideRight
                    font.pixelSize: model.selected ? 12 : 10
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignLeft
                    color: model.selected ? "#202020" : "#909090"
                    clip: true
                }

                Basic.Button {
                    Layout.alignment: Qt.AlignVCenter
                    icon.source: "../assets/icons/delete_icon.png"
                    icon.width: 21
                    icon.height: 21
                    background: Rectangle {
                        color: "transparent"
                    }
                    ToolTip.text: "Delete image"
                    ToolTip.visible: hovered
                    visible: model.selected
                    onClicked: projectController.delete_selected_thumbnail(row)
                }
            }

        }
    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            lblNoImages.visible = imgThumbnailModel.rowCount() <= 0;
            tblImgThumbs.visible = imgThumbnailModel.rowCount() > 0;
            tblImgThumbs.enabled = !mainController.is_task_running();
        }

        function onUpdateProgressSignal(val, msg) {
            tblImgThumbs.enabled = !mainController.is_task_running();
        }

        function onTaskTerminatedSignal(success_val, msg_data) {
            tblImgThumbs.enabled = !mainController.is_task_running();
        }

    }

    Connections {
        target: projectController

        function onProjectOpenedSignal(name) {
            lblNoImages.text = "No images to show!\nPlease import image(s).";
            tblImgThumbs.visible = imgThumbnailModel.rowCount() > 0;
        }

    }

}
