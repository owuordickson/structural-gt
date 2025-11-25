import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Theme 1.0
import "../widgets"

Dialog {
    id: dialogExtractGraph
    //parent: mainWindow
    anchors.centerIn: parent
    title: "Graph Extraction Options"
    modal: true
    width: 300
    height: 400


    //ColumnLayout {
    //    anchors.fill: parent

    //ScrollView {
    //   width: parent.width
    //   height: parent.height
    //Layout.alignment: Qt.AlignTop

    ColumnLayout {
        anchors.fill: parent

        GraphExtractWidget {
        }

        RowLayout {
            spacing: 10
            Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom

            Button {
                Layout.preferredWidth: 54
                Layout.preferredHeight: 30
                text: ""
                onClicked: dialogExtractGraph.close()

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
                id: btnRunExtractGraph
                Layout.preferredWidth: 40
                Layout.preferredHeight: 30
                text: ""
                visible: imageController.enable_img_controls()
                onClicked: {
                    dialogExtractGraph.close();
                    graphController.run_extract_graph();
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
    //}
    //}


    Connections {
        target: mainController

        function onImageChangedSignal() {
            btnRunExtractGraph.visible = imageController.enable_img_controls();
        }

    }

}