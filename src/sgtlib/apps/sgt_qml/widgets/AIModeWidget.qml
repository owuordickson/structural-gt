import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: aiModeControls
    Layout.preferredHeight: 40
    Layout.fillWidth: true
    Layout.alignment: Qt.AlignHCenter
    visible: mainController.display_image()
    enabled: mainController.enable_img_controls()

    RowLayout {
        id: aiModeContainer
        spacing: 2
        Layout.topMargin: 10
        Layout.bottomMargin: 5

        Label {
            id: lblAIMode
            text: "AI Mode"
            color: "#d0d0d0"
        }

        Switch {
            id: toggleAIMode
            checked: false
            onCheckedChanged: {
                if (checked) {
                    // Actions when switched on
                    lblAIMode.color = "#2266ff";
                    console.log("AI filter agent activated!");
                } else {
                    // Actions when switched off
                    lblAIMode.color = "#d0d0d0";
                    console.log("AI filter agent deactivated!");
                }
            }
        }

        BusyIndicator {
            id: progressAIMode
            running: toggleAIMode.checked
            width: 36
            height: 36
        }
    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            aiModeControls.visible = false; //mainController.display_image();
            aiModeControls.enabled = mainController.enable_img_controls();
        }
    }
}