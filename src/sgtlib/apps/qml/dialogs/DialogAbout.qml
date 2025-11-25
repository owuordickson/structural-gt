import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Theme 1.0

Dialog {
    id: dialogAbout
    //parent: mainWindow
    title: "About This Software"
    modal: true
    standardButtons: Dialog.Ok
    anchors.centerIn: parent
    width: 436
    height: 640
    background: Rectangle {color: Theme.background}

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true  // Ensures contents are clipped to the scroll view bounds


            Label {
                width: parent.width
                color: Theme.text
                //Layout.alignment: Qt.AlignHCenter
                property string aboutText: projectController.get_about_details()
                text: aboutText
                wrapMode: Text.WordWrap
                textFormat: Text.RichText  // Enable HTML formatting
                //maximumLineCount: 10  // Optional: Limits lines to avoid excessive height
                //elide: Text.ElideRight   // Optional: Adds "..." if text overflows
                onLinkActivated: (link) => Qt.openUrlExternally(link)  // Opens links in default browser
            }
        }
    }
}