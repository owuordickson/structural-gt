import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Rectangle {
    color: "#f0f0f0"
    border.color: "#c0c0c0"
    Layout.fillWidth: true
    Layout.fillHeight: true

    ScrollView {
        id: scrollViewImgProps
        anchors.fill: parent
        clip: true

        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff // Disable horizontal scrolling
        ScrollBar.vertical.policy: ScrollBar.AsNeeded // Enable vertical scrolling only when needed

        contentHeight: colImgPropsLayout.implicitHeight

        ColumnLayout {
            id: colImgPropsLayout
            width: scrollViewImgProps.width // Ensures it never exceeds parent width
            Layout.preferredWidth: parent.width // Fills the available width

            Text {
                text: "Image Properties"
                font.pixelSize: 12
                font.bold: true
                Layout.topMargin: 10
                Layout.bottomMargin: 5
                Layout.alignment: Qt.AlignHCenter
            }

            ImagePropertyWidget{}

            Rectangle {
                height: 1
                color: "#d0d0d0"
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 20
                Layout.leftMargin: 20
                Layout.rightMargin: 20
            }
            Text {
                text: "Graph Properties"
                font.pixelSize: 12
                font.bold: true
                Layout.topMargin: 10
                Layout.bottomMargin: 5
                Layout.alignment: Qt.AlignHCenter
            }

            Label {
                id: lblNoGraphProps
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 20
                text: "No properties to show!"
                color: "#808080"
                visible: graphPropsTableModel.rowCount() > 0 ? false : true
            }
            GraphPropertyWidget{}


            Rectangle {
                height: 1
                color: "#d0d0d0"
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 20
                Layout.leftMargin: 20
                Layout.rightMargin: 20
            }
            Text {
                text: "Microscopy Properties"
                font.pixelSize: 12
                font.bold: true
                Layout.topMargin: 10
                Layout.bottomMargin: 5
                Layout.alignment: Qt.AlignHCenter
            }

            MicroscopyPropertyWidget{}
        }
    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            lblNoGraphProps.visible = graphPropsTableModel.rowCount() > 0 ? false : true;
        }

        function onTaskTerminatedSignal(success_val, msg_data){
            lblNoGraphProps.visible = graphPropsTableModel.rowCount() > 0 ? false : true;
        }

    }

}
