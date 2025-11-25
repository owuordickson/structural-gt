import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Basic as Basic
import QtQuick.Controls.Imagine as Imagine
import Theme 1.0

// Icons retrieved from Iconfinder.com and used under the CC0 1.0 Universal Public Domain Dedication.
// Icons retrieved from https://www.flaticon.com and used under the CC0 1.0 Universal Public Domain Dedication.

Rectangle {
    id: statusBar
    width: parent.width
    height: 72
    color: Theme.background
    border.color: Theme.lightGray

    ColumnLayout {
        anchors.fill: parent
        spacing: 2

        // First Row: Progress Bar
        RowLayout {
            Layout.fillWidth: true // Make the row take the full width of the column
            Layout.leftMargin: 36
            Layout.rightMargin: 36 // Progressbar covers 80% of the width
            spacing: 5

            Imagine.ProgressBar {
                id: progressBar
                Layout.fillWidth: true
                visible: mainController.is_task_running()
                value: 0 // Example value (50% progress)
                from: 0
                to: 100
            }

            Basic.Button {
                id: btnCancel
                text: ""
                icon.source: "../assets/icons/cancel_icon.png"
                icon.width: 28
                icon.height: 28
                icon.color: "transparent"   // important for PNGs
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Cancel task!"
                ToolTip.visible: btnCancel.hovered
                visible: mainController.is_task_running()
                enabled: mainController.is_task_running()
                onClicked: {
                    btnCancel.visible = false;
                    lblStatusMsg.text = "initiating abort...";
                    mainController.stop_current_task(1);
                }
            }
        }

        // Second Row: Label and Button
        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 36
            Layout.rightMargin: 36
            Layout.bottomMargin: 10
            spacing: 5

            Label {
                id: lblVersion
                Layout.alignment: Qt.AlignLeft
                text: projectController.get_sgt_version()
                visible: !mainController.is_task_running()
                Layout.fillWidth: true
                color: Theme.blue
            }

            Label {
                id: lblStatusMsg
                Layout.alignment: Qt.AlignLeft
                text: "Please wait..."
                visible: mainController.is_task_running()
                Layout.fillWidth: true
                color: Theme.blue
            }

            Basic.Button {
                id: btnNotify
                text: ""
                icon.source: "../assets/icons/notify_icon.png"
                icon.width: 21
                icon.height: 21
                background: Rectangle { color: "transparent" }
                ToolTip.text: "Check for updates"
                ToolTip.visible: btnNotify.hovered
                onClicked: drpDownNotify.open()
                enabled: true
                visible: !mainController.is_task_running()

                Popup {
                    id: drpDownNotify
                    width: 128
                    height: 64
                    modal: false
                    focus: true
                    x: -60
                    y: -60
                    background: Rectangle {
                        color: Theme.background
                        border.color: Theme.lightGray
                        border.width: 1
                        radius: 2
                    }

                    ColumnLayout {
                        anchors.fill: parent

                        Label {
                            id: lblNotifyMsg
                            font.pixelSize: 10
                            wrapMode: Text.Wrap
                            textFormat: Text.RichText  // Enable HTML formatting
                            onLinkActivated: (link) => Qt.openUrlExternally(link)  // Opens links in default browser
                            text: projectController.get_software_download_details()
                        }
                    }

                }

            }
        }
    }

    Connections {
        target: mainController

        function onUpdateProgressSignal(val, msg) {
            if (val <= 100) {
                progressBar.value = val;
            } else {
                progressBar.value = 50;
            }
            lblStatusMsg.text = msg;
            lblStatusMsg.color = Theme.darkGreen;

            lblVersion.visible = !mainController.is_task_running();
            lblStatusMsg.visible = mainController.is_task_running();
            progressBar.visible = mainController.is_task_running();
            btnCancel.visible = mainController.is_task_running();
            btnNotify.visible = !mainController.is_task_running();
            btnCancel.enabled = mainController.is_task_running();
        }

        function onErrorSignal(msg) {
            progressBar.value = 0;
            lblStatusMsg.text = msg;
            lblStatusMsg.color = Theme.red;

            lblVersion.visible = !mainController.is_task_running();
            lblStatusMsg.visible = mainController.is_task_running();
            progressBar.visible = mainController.is_task_running();
            btnCancel.visible = mainController.is_task_running();
            btnNotify.visible = !mainController.is_task_running();
            btnCancel.enabled = mainController.is_task_running();
        }

        function onTaskTerminatedSignal(success_val, msg_data) {
            if (success_val) {
                lblStatusMsg.color = Theme.blue;
                //lblStatusMsg.text = "Please wait...";
            } else {
                lblStatusMsg.color = Theme.red;
                lblStatusMsg.text = "Task terminated due to an error. Try again.";
            }

            if (msg_data.length > 0) {
                dialogAlert.title = msg_data[0];
                dialogAlert.alertMsg = msg_data[1];
                dialogAlert.alertColor = success_val ? Theme.blue : Theme.red;
                dialogAlert.open();
            }

            const updates_available = projectController.check_for_updates();
            lblNotifyMsg.text = projectController.get_software_download_details();
            if (updates_available) {
                btnNotify.icon.source = "../assets/icons/notify_active_icon.png";
                btnNotify.icon.width = 28;
                btnNotify.icon.height = 28;
            } else {
                btnNotify.icon.source = "../assets/icons/notify_icon.png";
                btnNotify.icon.width = 21;
                btnNotify.icon.height = 21;
            }

            lblVersion.visible = !mainController.is_task_running();
            lblStatusMsg.visible = mainController.is_task_running();
            progressBar.visible = mainController.is_task_running();
            btnCancel.visible = mainController.is_task_running();
            btnNotify.visible = !mainController.is_task_running();
            btnCancel.enabled = mainController.is_task_running();
        }

    }
}
