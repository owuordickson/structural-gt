import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: projectControls // used for external access
    Layout.fillWidth: true
    Layout.fillHeight: true

    property int tblRowHeight: 50


    ColumnLayout {
        id: projectCtrlLayout
        anchors.fill: parent
        spacing: 10

        Label {
            id: lblNoImages
            Layout.alignment: Qt.AlignHCenter
            Layout.topMargin: 20
            text: "No images to show!\nPlease import/add image."
            color: "#808080"
            visible: imgListTableModel.rowCount() > 0 ? false : true
            //visible: true
        }

        // image list
        // image settings and options (saved from selection) - linked to image list

        TableView {
            id: tableView
            height: imgListTableModel.rowCount() * tblRowHeight
            //width: 300
            Layout.fillWidth: true
            Layout.topMargin: 5
            Layout.leftMargin: 2
            Layout.rightMargin: 2
            model: imgListTableModel
            visible: imgListTableModel.rowCount() > 0 ? true : false
            //visible: false

            delegate: Rectangle {
                implicitWidth: tableView.width
                implicitHeight: tblRowHeight
                color: row % 2 === 0 ? "#f5f5f5" : "#ffffff" // Alternating colors

                RowLayout {

                    Rectangle {
                        width: tblRowHeight
                        height: tblRowHeight
                        radius: 4
                        border.width: 1
                        border.color: "black"

                        Image {
                            id: imgThumbnail
                            anchors.fill: parent
                            source: ""
                        }

                    }

                    Label {
                        id: lblImgItem
                        text: display
                    }
                }


            }


        }

    }

}
