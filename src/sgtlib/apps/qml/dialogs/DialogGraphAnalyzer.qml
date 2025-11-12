import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Dialog {
    id: dialogRunAnalyzer
    anchors.centerIn: parent
    title: "Select Graph Computations"
    modal: true
    width: 264
    height: 560

    ColumnLayout {
        anchors.fill: parent

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true  // Ensures contents are clipped to the scroll view bounds

            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff // Disable horizontal scrolling
            ScrollBar.vertical.policy: ScrollBar.AsNeeded // Enable vertical scrolling only when needed

            GTWidget {
            }
        }

        RowLayout {
            spacing: 10
            Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom

            Button {
                Layout.preferredWidth: 54
                Layout.preferredHeight: 30
                text: ""
                onClicked: dialogRunAnalyzer.close()

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: "#bc0000"

                    Label {
                        text: "Cancel"
                        color: "#ffffff"
                        anchors.centerIn: parent
                    }
                }
            }

            Button {
                Layout.preferredWidth: 40
                Layout.preferredHeight: 30
                text: ""
                onClicked: {
                    dialogRunAnalyzer.close();
                    graphController.run_graph_analyzer();
                }

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: "#22bc55"

                    Label {
                        text: "OK"
                        color: "#ffffff"
                        anchors.centerIn: parent
                    }
                }
            }
        }
    }
}