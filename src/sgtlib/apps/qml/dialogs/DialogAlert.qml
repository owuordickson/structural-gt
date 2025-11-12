import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Dialog {
    id: dialogAlert
    //parent: mainWindow
    title: ""
    modal: true
    standardButtons: Dialog.Ok
    anchors.centerIn: parent
    width: 300
    height: 150

    /*contentItem: ColumnLayout {
            spacing: 10
            width: parent.width

            // Custom Header for the Dialog Title
            Label {
                text: dialogAlert.title
                font.bold: true
                font.pointSize: 14
                color: "red"  // Change title color to red
                horizontalAlignment: Text.AlignHCenter
                Layout.fillWidth: true  // Ensure it spans the full width
            }

            Label {
                id: lblAlertMsg
                width: parent.width
                wrapMode: Text.Wrap  // Enable text wrapping
                anchors.centerIn: parent
                leftPadding: 10
                rightPadding: 10
                horizontalAlignment: Text.AlignJustify  // Justify the text
                color: "#bc2222"
                text: ""
            }
        }*/

    Label {
        id: lblAlertMsg
        width: parent.width
        wrapMode: Text.Wrap  // Enable text wrapping
        anchors.centerIn: parent
        leftPadding: 10
        rightPadding: 10
        horizontalAlignment: Text.AlignJustify  // Justify the text
        color: "#bc2222"
        text: ""
    }
}