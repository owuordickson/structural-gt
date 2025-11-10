import QtQuick
import QtQuick.Controls
import QtQuick.Layouts


Rectangle {
    id: imgViewControls
    height: 32
    Layout.fillHeight: false
    Layout.fillWidth: true
    color: "transparent"
    visible: imageController.display_image()

    RowLayout {
        anchors.verticalCenter: parent.verticalCenter
        anchors.horizontalCenter: parent.horizontalCenter

        Switch {
            id: toggleShowGiantGraph
            visible: graphController.display_graph()
            text: "only giant"
            ToolTip.text: "Display only the giant graph network."
            ToolTip.visible: toggleShowGiantGraph.hovered
            checked: false // Initial state
            onCheckedChanged: {
                if (checked) {
                    // Actions when switched on
                    graphController.reload_graph_image(true);
                } else {
                    // Actions when switched off
                    graphController.reload_graph_image(false);
                }
            }
        }

        Button {
            id: btnLoad3DGraph
            leftPadding: 10
            rightPadding: 10
            text: " view"
            icon.source: "../assets/icons/3d_icon.png"
            icon.width: 21
            icon.height: 21
            icon.color: "transparent"   // important for PNGs
            ToolTip.text: "Load OVITO 3D graph visualization."
            ToolTip.visible: btnLoad3DGraph.hovered
            visible: graphController.display_graph()
            onClicked: graphController.load_graph_simulation()
        }

        Button {
            id: btnGraphRating
            leftPadding: 10
            rightPadding: 10
            text: " rate"
            icon.source: "../assets/icons/thumbs-up-emoji.png"
            icon.width: 21
            icon.height: 21
            icon.color: "transparent"   // important for PNGs
            ToolTip.text: "How good is the graph? Give a score..."
            ToolTip.visible: btnGraphRating.hovered
            visible: graphController.display_graph()
            onClicked: drpDownRating.open()

            Popup {
                id: drpDownRating
                width: 400
                height: 172
                modal: true
                focus: false
                x: -225
                y: 32
                background: Rectangle {
                    color: "#f0f0f0"
                    border.color: "#d0d0d0"
                    border.width: 1
                    radius: 2
                }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 2

                    GraphRatingWidget {
                        id: graphRating
                    }

                    RowLayout {
                        spacing: 10
                        Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom

                        Button {
                            Layout.preferredWidth: 54
                            Layout.preferredHeight: 30
                            text: ""
                            onClicked: drpDownRating.close()

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
                            id: btnSendRating
                            Layout.preferredWidth: 40
                            Layout.preferredHeight: 30
                            text: ""
                            visible: imageController.enable_img_controls()
                            onClicked: {
                                drpDownRating.close();
                                graphController.rate_graph(graphRating.rating);
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
        }
    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            imgViewControls.visible = imageController.display_image();
            toggleShowGiantGraph.visible = graphController.display_graph();
            btnLoad3DGraph.visible = graphController.display_graph();
            btnGraphRating.visible = graphController.display_graph();
            btnSendRating.visible = imageController.enable_img_controls();
        }

    }

}


