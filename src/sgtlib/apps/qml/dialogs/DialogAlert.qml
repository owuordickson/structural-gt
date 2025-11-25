import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Theme 1.0

Dialog {
    id: dialogAlert
    //parent: mainWindow
    title: ""
    modal: true
    standardButtons: Dialog.Ok
    anchors.centerIn: parent
    width: 300
    height: 150
    background: Rectangle {color: Theme.background}

    property string alertMsg: ""
    property string alertColor: Theme.red

    Label {
        width: parent.width
        wrapMode: Text.Wrap  // Enable text wrapping
        anchors.centerIn: parent
        leftPadding: 10
        rightPadding: 10
        horizontalAlignment: Text.AlignJustify  // Justify the text
        color: dialogAlert.alertColor
        text: dialogAlert.alertMsg
    }

    Connections {
        target: mainController

        function onShowAlertSignal(title, msg) {
            dialogAlert.title = title;
            dialogAlert.alertMsg = msg;
            dialogAlert.alertColor = Theme.blue;
            dialogAlert.open();
        }
    }
}