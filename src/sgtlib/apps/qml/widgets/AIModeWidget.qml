import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Effects
import QtQuick.Controls.Basic as Basic

Rectangle {
    id: aiModeControls
    width: parent.width - 10
    Layout.alignment: Qt.AlignTop
    height: 72
    radius: 5
    color: "#f0fff0"
    Layout.margins: 5   // shorthand for top/left/right/bottom
    visible: imageController.display_image()
    enabled: imageController.enable_img_controls()
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
            spacing: 1
            //Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter

            Label {
                id: lblAIMode
                text: "AI Mode"
                color: "#d0d0d0"
            }

            Switch {
                id: toggleAIMode
                ToolTip.text: "Activate"
                ToolTip.visible: toggleAIMode.hovered
                checked: aiController.ai_mode_active
                onCheckedChanged: {
                    if (checked) {
                        lblAIMode.color = "#2266ff";
                        toggleAIMode.ToolTip.text = "Deactivate";
                        aiController.toggle_ai_mode(true);
                        aiController.run_ai_filter_search();
                    } else {
                        lblAIMode.color = "#d0d0d0";
                        toggleAIMode.ToolTip.text = "Activate";
                        aiController.toggle_ai_mode(false);
                    }
                }
            }

            Basic.Button {
                id: btnRunAI
                text: ""
                icon.source: "../assets/icons/replay_icon.png"
                icon.width: 28
                icon.height: 28
                icon.color: "transparent"   // important for PNGs
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Re-run"
                ToolTip.visible: btnRunAI.hovered
                visible: !aiController.ai_busy && aiController.ai_mode_active
                onClicked: aiController.reset_ai_filter_results()
            }

            Column {
                visible: aiController.ai_busy
                SpinnerProgress {
                    running: aiController.ai_busy
                    width: 24
                    height: 24
                }
            }

            Basic.Button {
                id: btnStopAI
                text: ""
                icon.source: "../assets/icons/stop_icon.png"
                icon.width: 24
                icon.height: 24
                icon.color: "transparent"   // important for PNGs
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Stop!"
                ToolTip.visible: btnStopAI.hovered
                visible: aiController.ai_busy
                onClicked: mainController.stop_current_task(2)
            }
        }

        AISearchWidget {
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
            Layout.bottomMargin: 5
            visible: aiController.ai_busy

            Label {
                id: lblAIStatusMsg
                color: "#909090"
                text: "please wait..."
            }
        }
    }

    Connections {
        target: aiController

        function onUpdateAIProgressSignal(val, msg) {
            if (val <= 100) {
                lblAIStatusMsg.text = val + "%: " + msg;
            } else {
                lblAIStatusMsg.text = msg;
            }
        }
    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            aiModeControls.visible = imageController.display_image();
            aiModeControls.enabled = imageController.enable_img_controls();
        }
    }
}