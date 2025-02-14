import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Icons retrieved from Iconfinder.com and used under the CC0 1.0 Universal Public Domain Dedication.

Rectangle {
    id: statusBar
    width: parent.width
    height: 72
    color: "#f0f0f0"
    border.color: "#d0d0d0"

    ColumnLayout {
        anchors.fill: parent
        spacing: 2

        // First Row: Progress Bar
        RowLayout {
            Layout.fillWidth: true // Make the row take full width of the column
            Layout.alignment: Qt.AlignLeft
            Layout.leftMargin: 36
            Layout.rightMargin: 36 // Progress bar covers 80% of the width
            spacing: 5

            ProgressBar {
                id: progressBar
                Layout.fillWidth: true
                //visible: false
                value: 0 // Example value (50% progress)
                from: 0
                to: 100
            }


            Button {
                id: btnCancel
                text: ""
                ToolTip.text: "Cancel task!"
                ToolTip.visible: btnCancel.hovered
                background: transientParent
                icon.source: "../assets/icons/cancel_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                //visible: false
                enabled: false
                onClicked: {
                    progressBar.value = 0 // Reset progress
                    console.log("Progress canceled")
                }
            }
        }

        // Second Row: Label and Button
        Label {
            id: lblStatusMsg
            Layout.alignment: Qt.AlignLeft
            Layout.leftMargin: 36
            Layout.rightMargin: 36 // Progress bar covers 80% of the width
            Layout.bottomMargin: 10
            text: "v3.0.1"  // Copyright (C) 2024, the Regents of the University of Michigan.
            Layout.fillWidth: true
            color: "blue"
        }
    }
}
