import QtQuick
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15 as MaterialControls
import QtQuick.Layouts
//import QtQuick.Controls.Basic as Basic
import Theme 1.0
import "tabs"

Rectangle {
    width: parent.width
    height: parent.height
    color: Theme.background
    border.color: Theme.borderColor

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
                    color: parent.checked ? Theme.blueText : Theme.whiteText
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
                    color: parent.checked ? Theme.blueText : Theme.whiteText
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
                    color: parent.checked ? Theme.blueText
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
