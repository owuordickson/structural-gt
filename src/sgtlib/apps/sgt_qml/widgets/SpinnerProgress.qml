import QtQuick
import QtQuick.Controls.Basic as Basic


Basic.BusyIndicator {
    id: control

    // Allow user to override width/height, default to 64
    property int defaultSize: 64
    width: (control.implicitWidth > 0 ? control.implicitWidth : defaultSize)
    height: (control.implicitHeight > 0 ? control.implicitHeight : defaultSize)

    anchors.horizontalCenter: parent ? parent.horizontalCenter : undefined

    property double markerSize: control.width > 48 ? 5 : 2.5

    contentItem: Item {
        implicitWidth: control.defaultSize
        implicitHeight: control.defaultSize

        Item {
            id: item
            anchors.centerIn: parent
            width: control.width
            height: control.height
            opacity: control.running ? 1 : 0

            Behavior on opacity {
                OpacityAnimator { duration: 250 }
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
                    implicitWidth: markerSize * 2
                    implicitHeight: markerSize * 2
                    radius: 5
                    color: "#2299ff"

                    required property int index

                    transform: [
                        Translate {
                            y: -Math.min(item.width, item.height) * 0.5 + markerSize
                        },
                        Rotation {
                            angle: delegate.index / repeater.count * 360
                            origin.x: markerSize
                            origin.y: markerSize
                        }
                    ]
                }
            }
        }
    }
}
