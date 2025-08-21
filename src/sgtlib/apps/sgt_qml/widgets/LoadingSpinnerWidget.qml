import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Item {
    id: waitOverlay
    anchors.fill: parent
    visible: mainController.wait
    z: 9999

    Rectangle {
        anchors.fill: parent
        color: "#80000000" // semi-transparent dark
    }

    Column {
        anchors.centerIn: parent
        spacing: 12

        Basic.BusyIndicator {
            id: control
            running: mainController.wait
            width: 64
            height: 64
            anchors.horizontalCenter: parent.horizontalCenter

            contentItem: Item {
                implicitWidth: 64
                implicitHeight: 64

                Item {
                    id: item
                    x: parent.width / 2 - 32
                    y: parent.height / 2 - 32
                    width: 64
                    height: 64
                    opacity: control.running ? 1 : 0

                    Behavior on opacity {
                        OpacityAnimator {
                            duration: 250
                        }
                    }

                    RotationAnimator {
                        target: item
                        running: control.visible && control.running
                        from: 0
                        to: 360
                        loops: Animation.Infinite
                        duration: 1250
                    }

                    Repeater {
                        id: repeater
                        model: 6

                        Rectangle {
                            id: delegate
                            x: item.width / 2 - width / 2
                            y: item.height / 2 - height / 2
                            implicitWidth: 10
                            implicitHeight: 10
                            radius: 5
                            color: "#2299ff"

                            required property int index

                            transform: [
                                Translate {
                                    y: -Math.min(item.width, item.height) * 0.5 + 5
                                },
                                Rotation {
                                    angle: delegate.index / repeater.count * 360
                                    origin.x: 5
                                    origin.y: 5
                                }
                            ]
                        }
                    }
                }
            }
        }

        Label {
            text: mainController.wait_text
            font.pointSize: 21
            color: "#2299ff"
            horizontalAlignment: Text.AlignHCenter
            anchors.horizontalCenter: parent.horizontalCenter
        }
    }
}