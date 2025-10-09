import QtQuick
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15 as MaterialControls
import QtQuick.Layouts
//import QtQuick.Controls.Basic as Basic
import "components"

Rectangle {
    width: parent.width
    height: parent.height
    color: "#f0f0f0"
    border.color: "#c0c0c0"

    ColumnLayout {
        anchors.fill: parent

        MaterialControls.TabBar {
            id: tabBar
            currentIndex: 2
            Layout.fillWidth: true

            TabButton {
                text: "Project"
                contentItem: Text {
                    text: parent.text
                    font: parent.font
                    color: parent.checked ? "#2266ff" : "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                text: "Properties"
                contentItem: Text {
                    text: parent.text
                    font: parent.font
                    color: parent.checked ? "#2266ff" : "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                text: "Filters"
                contentItem: Text {
                    text: parent.text
                    font: parent.font
                    color: parent.checked ? "#2266ff" : "white" // #E91E63
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }
        }

        StackLayout {
            id: stackLayout
            //width: parent.width
            Layout.fillWidth: true
            currentIndex: tabBar.currentIndex


            ProjectNav {
            }

            ImageProperties {
            }

            ImageFilters {
            }


        }
    }

    Connections {
        target: projectController

        function onProjectOpenedSignal(name) {
            tabBar.currentIndex = 0;
        }
    }
}
