import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Effects
//import QtQuick.Controls.Basic as Basic

Rectangle {
    id: aiModeControls
    width: parent.width - 10
    height: 72
    radius: 5
    color: "#f0fff0"
    Layout.topMargin: 5
    Layout.leftMargin: 5
    Layout.rightMargin: 5
    visible: mainController.display_image()
    enabled: mainController.enable_img_controls()

    layer.enabled: true
    layer.effect: MultiEffect {
        anchors.fill: parent
        shadowEnabled: true
        shadowColor: "#80000000"
        shadowBlur: 0.3      // 0.0 - 1.0
        shadowHorizontalOffset: 0
        shadowVerticalOffset: 8
    }


    ColumnLayout {
        anchors.fill: parent
        Layout.alignment: Qt.AlignHCenter

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
            Layout.bottomMargin: 10
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