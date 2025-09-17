import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
//import QtQuick.Controls.Basic as Basic

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
        //Layout.topMargin: 5
        //Layout.bottomMargin: 10

        Label {
            id: lblAIMode
            text: "AI Mode"
            color: "#d0d0d0"
        }

        Switch {
            id: toggleAIMode
            checked: mainController.ai_mode_active
            onCheckedChanged: {
                if (checked) {
                    // Actions when switched on
                    lblAIMode.color = "#2266ff";
                    mainController.toggle_ai_mode(true);
                } else {
                    // Actions when switched off
                    lblAIMode.color = "#d0d0d0";
                    mainController.toggle_ai_mode(false);
                }
            }
        }

        BusyIndicator {
            id: progressAIMode
            running: toggleAIMode.checked
            width: 32
            height: 32
            //antialiasing: true
        }
    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            aiModeControls.visible = mainController.display_image();
            aiModeControls.enabled = mainController.enable_img_controls();
        }
    }
}