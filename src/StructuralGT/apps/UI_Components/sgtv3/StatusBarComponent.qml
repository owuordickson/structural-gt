import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: statusBar
    width: parent.width
    height: 60
    color: "#f0f0f0"
    border.color: "#d0d0d0"

    ColumnLayout {
        anchors.fill: parent
        anchors.centerIn: parent
        spacing: 2

        // First Row: Progress Bar
        ProgressBar {
            id: progressBar
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
            Layout.leftMargin: 36
            Layout.rightMargin: 36 // Progress bar covers 80% of the width
            visible: false
            value: 0 // Example value (50% progress)
            from: 0
            to: 100
        }

        // Second Row: Label and Button
        RowLayout {
            Layout.fillWidth: true // Make the row take full width of the column
            Layout.alignment: Qt.AlignLeft
            Layout.leftMargin: 36
            Layout.rightMargin: 36 // Progress bar covers 80% of the width
            //Layout.alignment: Qt.AlignHCenter
            spacing: 10

            Label {
                id: lblStatusMsg
                text: "Welcome..."
                Layout.fillWidth: true
            }

            Button {
                id: btnCancel
                text: "Cancel"
                visible: false
                enabled: false
                onClicked: {
                    progressBar.value = 0 // Reset progress
                    console.log("Progress canceled")
                }
            }
        }
    }
}
