import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Effects
import QtQuick.Controls.Basic as Basic

Rectangle {
    id: aiModeControls
    width: parent.width - 10
    height: 72
    radius: 5
    color: "#f0fff0"
    Layout.margins: 5   // shorthand for top/left/right/bottom
    visible: mainController.display_image()
    enabled: mainController.enable_img_controls()

    layer.enabled: true
    layer.effect: MultiEffect {
        shadowEnabled: true
        shadowColor: "#80000000"
        shadowBlur: 0.3
        shadowHorizontalOffset: 0
        shadowVerticalOffset: 2
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 4

        RowLayout {
            id: aiModeContainer
            spacing: 2
            //Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

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
                        lblAIMode.color = "#2266ff";
                        mainController.toggle_ai_mode(true);
                        mainController.run_ai_filter_search();
                    } else {
                        lblAIMode.color = "#d0d0d0";
                        mainController.toggle_ai_mode(false);
                    }
                }
            }

            Basic.Button {
                id: btnRunAI
                text: ""
                Layout.preferredWidth: 36
                Layout.preferredHeight: 36
                icon.source: "../assets/icons/play_icon.png"
                icon.width: 24
                icon.height: 24
                icon.color: "transparent"   // important for PNGs
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Re-run AI search."
                ToolTip.visible: btnRunAI.hovered
                visible: !mainController.ai_busy && mainController.ai_mode_active
                onClicked: console.log("Run AI")
            }

            BusyIndicator {
                id: progressAIMode
                running: mainController.ai_busy
                width: 28
                height: 28
            }

            Basic.Button {
                id: btnStopAI
                text: ""
                Layout.preferredWidth: 36
                Layout.preferredHeight: 36
                icon.source: "../assets/icons/stop_icon.png"
                icon.width: 28
                icon.height: 28
                icon.color: "transparent"   // important for PNGs
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Stop the current AI task!"
                ToolTip.visible: btnStopAI.hovered
                visible: mainController.ai_busy
                onClicked: console.log("Stop AI task!")
            }
        }

        RowLayout {
            id: aiControls
            spacing: 6
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
            aiModeControls.visible = mainController.display_image();
            aiModeControls.enabled = mainController.enable_img_controls();
        }
    }
}