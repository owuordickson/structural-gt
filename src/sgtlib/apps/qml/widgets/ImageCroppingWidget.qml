import QtQuick
import QtQuick.Controls
import QtQuick.Shapes

Item {
    id: overlay
    anchors.fill: parent
    property alias rectX: cropRect.x
    property alias rectY: cropRect.y
    property alias rectWidth: cropRect.width
    property alias rectHeight: cropRect.height

    property color borderColor: "#00AEEF"
    property real handleSize: 24
    property real borderWidth: 2

    // ✅ The transparent rectangle overlay
    Rectangle {
        id: cropRect
        anchors.centerIn: parent
        width: parent.width - 100
        height: parent.height - 100
        color: "transparent"
        border.color: overlay.borderColor
        border.width: overlay.borderWidth
        z: 1
    }

    // ✅ Semi-transparent mask outside the crop area
    /*Rectangle {
        anchors.fill: parent
        color: "#80000000"
        z: 0
        layer.enabled: true
        layer.samplerName: "background"

        ShaderEffectSource {
            sourceItem: cropRect
            hideSource: true
        }

        // Clip transparent hole
        /*layer.effect: ShaderEffect {
            fragmentShader: "
                uniform lowp sampler2D background;
                uniform lowp sampler2D source;
                varying highp vec2 qt_TexCoord0;
                void main() {
                    lowp vec4 bg = texture2D(background, qt_TexCoord0);
                    lowp vec4 fg = texture2D(source, qt_TexCoord0);
                    gl_FragColor = mix(bg, vec4(0.0), fg.a);
                }"
        }
    }*/

    // ✅ Handles (arrows)
    Repeater {
        model: [
            { pos: "top", cursor: Qt.SizeVerCursor, x: 0.5, y: 0.0, icon: "▲" },
            { pos: "bottom", cursor: Qt.SizeVerCursor, x: 0.5, y: 1.0, icon: "▼" },
            { pos: "left", cursor: Qt.SizeHorCursor, x: 0.0, y: 0.5, icon: "◄" },
            { pos: "right", cursor: Qt.SizeHorCursor, x: 1.0, y: 0.5, icon: "►" }
        ]

        delegate: Rectangle {
            id: handle
            width: overlay.handleSize
            height: overlay.handleSize
            color: "transparent"
            border.color: "white"
            border.width: 1
            radius: 6
            anchors.centerIn: parent
            z: 2
            x: cropRect.x + (modelData.x * cropRect.width) - width / 2
            y: cropRect.y + (modelData.y * cropRect.height) - height / 2

            Text {
                anchors.centerIn: parent
                text: modelData.icon
                color: "white"
                font.pixelSize: 14
            }

            MouseArea {
                anchors.fill: parent
                cursorShape: modelData.cursor
                drag.target: cropRect
                drag.axis: modelData.pos === "top" || modelData.pos === "bottom" ? Drag.YAxis : Drag.XAxis

                property real startX
                property real startY
                onPressed: {
                    startX = cropRect.x
                    startY = cropRect.y
                }

                onPositionChanged: (mouse) => {
                    if (modelData.pos === "top") {
                        let dy = mouse.y
                        cropRect.y += dy
                        cropRect.height -= dy
                    } else if (modelData.pos === "bottom") {
                        cropRect.height = mouse.y
                    } else if (modelData.pos === "left") {
                        let dx = mouse.x
                        cropRect.x += dx
                        cropRect.width -= dx
                    } else if (modelData.pos === "right") {
                        cropRect.width = mouse.x
                    }
                }
            }
        }
    }
}