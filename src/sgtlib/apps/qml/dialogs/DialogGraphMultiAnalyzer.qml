import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Theme 1.0
import "../widgets"

Dialog {
    id: dialogRunMultiAnalyzer
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
                onClicked: dialogRunMultiAnalyzer.close()

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: Theme.errorColor

                    Label {
                        text: "Cancel"
                        color: Theme.whiteText
                        anchors.centerIn: parent
                    }
                }
            }

            Button {
                Layout.preferredWidth: 40
                Layout.preferredHeight: 30
                text: ""
                onClicked: {
                    dialogRunMultiAnalyzer.close();
                    graphController.run_multi_graph_analyzer();
                }

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: Theme.successColor

                    Label {
                        text: "OK"
                        color: Theme.whiteText
                        anchors.centerIn: parent
                    }
                }
            }
        }
    }
}