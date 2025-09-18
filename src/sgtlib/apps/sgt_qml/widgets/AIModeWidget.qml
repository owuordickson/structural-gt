import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
//import QtQuick.Controls.Basic as Basic

ColumnLayout {
    id: aiModeControls
    Layout.preferredHeight: 64
    Layout.preferredWidth: parent.width
    Layout.alignment: Qt.AlignHCenter
    visible: mainController.display_image()
    enabled: mainController.enable_img_controls()

    RowLayout {
        id: aiModeContainer
        spacing: 2
        Layout.fillWidth: true
        Layout.alignment: Qt.AlignHCenter
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
                    mainController.run_ai_filter_search();
                } else {
                    // Actions when switched off
                    lblAIMode.color = "#d0d0d0";
                    mainController.toggle_ai_mode(false);
                }
            }
        }

        BusyIndicator {
            id: progressAIMode
            running: mainController.ai_busy
            width: 32
            height: 32
            //antialiasing: true
        }
    }

    RowLayout {
        id: aiControls
        spacing: 2
        Layout.fillWidth: true
        Layout.alignment: Qt.AlignHCenter

        CheckBox {
            id: cbxFilters
            text: "Estimate Values"
            checked: true
        }

        CheckBox {
            id: cbxBrightness
            text: "Brightness/Contrast"
            checked: false
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