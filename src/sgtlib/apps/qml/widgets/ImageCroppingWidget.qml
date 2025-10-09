import QtQuick
import QtQuick.Controls

Item {
    id: overlay
    anchors.fill: parent

    // rectangle coordinates (px, relative to overlay)
    property real leftPt: 0
    property real topPt: 0
    property real rightPt: width
    property real bottomPt: height

    // helpers / config
    property int handleSize: 24
    property int minWidth: 40
    property int minHeight: 40
    property color borderColor: "#2266ff"
    property bool _initialized: false

    // snapshots used while dragging
    property real _startLeft: 0
    property real _startTop: 0
    property real _startRight: 0
    property real _startBottom: 0
    property real _startMouseX: 0
    property real _startMouseY: 0

    // init to full size once size is known
    onWidthChanged: {
        if (!_initialized && width > 0 && height > 0) {
            leftPt = 0;
            topPt = 0;
            rightPt = width;
            bottomPt = height;
            _initialized = true;
        }
    }

    // visible crop rectangle (computed from coords)
    Rectangle {
        id: cropRect
        x: overlay.leftPt
        y: overlay.topPt
        width: Math.max(overlay.minWidth, overlay.rightPt - overlay.leftPt)
        height: Math.max(overlay.minHeight, overlay.bottomPt - overlay.topPt)
        color: "transparent"
        border.width: 2
        border.color: overlay.borderColor
        z: 2

        // move the whole rectangle by dragging inside
        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.SizeAllCursor
            onPressed: (mouse) => {
                // snapshot coords and mouse position (parent is cropRect here)
                overlay._startLeft = overlay.leftPt;
                overlay._startRight = overlay.rightPt;
                overlay._startTop = overlay.topPt;
                overlay._startBottom = overlay.bottomPt;
                overlay._startMouseX = parent.x + mouse.x; // mouse pos relative to overlay
                overlay._startMouseY = parent.y + mouse.y;
            }
            onPositionChanged: (mouse) => {
                var curX = parent.x + mouse.x;
                var curY = parent.y + mouse.y;
                var dx = curX - overlay._startMouseX;
                var dy = curY - overlay._startMouseY;

                // shift all coordinates while clamping to parent bounds
                var newLeft = overlay._startLeft + dx;
                var newRight = overlay._startRight + dx;
                var newTop = overlay._startTop + dy;
                var newBottom = overlay._startBottom + dy;

                // clamp horizontally
                var w = newRight - newLeft;
                if (newLeft < 0) {
                    newLeft = 0;
                    newRight = newLeft + w;
                }
                if (newRight > overlay.width) {
                    newRight = overlay.width;
                    newLeft = newRight - w;
                }
                // clamp vertically
                var h = newBottom - newTop;
                if (newTop < 0) {
                    newTop = 0;
                    newBottom = newTop + h;
                }
                if (newBottom > overlay.height) {
                    newBottom = overlay.height;
                    newTop = newBottom - h;
                }

                // apply
                overlay.leftPt = newLeft;
                overlay.rightPt = newRight;
                overlay.topPt = newTop;
                overlay.bottomPt = newBottom;
            }
        }
    }

    // helper function for clamp (optional; using Math.min/Math.max inline below)

    // --- Handles: top, bottom, left, right ---
    // Top handle (modifies `top` only; bottom is fixed)
    Rectangle {
        id: topHandle
        width: overlay.handleSize * 3
        height: overlay.handleSize
        x: (overlay.leftPt + overlay.rightPt) / 2 - width / 2
        y: overlay.topPt - height / 2
        color: "transparent"
        z: 3

        /*Text {
            anchors.centerIn: parent; text: "↕"; color: "white"; font.pixelSize: 18
        }*/

        Image {
            anchors.centerIn: parent
            source: "../assets/icons/top_down_arrow.png"
            width: 21
            height: 21
            opacity: 1.0
        }

        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.SizeVerCursor
            onPressed: (mouse) => {
                overlay._startTop = overlay.topPt;
                overlay._startBottom = overlay.bottomPt;
                overlay._startMouseY = parent.y + mouse.y; // global relative to overlay
            }
            onPositionChanged: (mouse) => {
                var curY = parent.y + mouse.y;
                var dy = curY - overlay._startMouseY;
                var newTop = overlay._startTop + dy;

                // clamp: 0 <= newTop <= bottom - minHeight
                newTop = Math.max(0, Math.min(newTop, overlay.bottomPt - overlay.minHeight));
                overlay.topPt = newTop;
            }
        }
    }

    // Bottom handle (modifies `bottom` only; top is fixed)
    Rectangle {
        id: bottomHandle
        width: overlay.handleSize * 3
        height: overlay.handleSize
        x: (overlay.leftPt + overlay.rightPt) / 2 - width / 2
        y: overlay.bottomPt - height / 2
        color: "transparent"
        z: 3

        /*Text {
            anchors.centerIn: parent; text: "↕"; color: "white"; font.pixelSize: 18
        }*/

        Image {
            anchors.centerIn: parent
            source: "../assets/icons/top_down_arrow.png"
            width: 21
            height: 21
            opacity: 1.0
        }

        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.SizeVerCursor
            onPressed: (mouse) => {
                overlay._startTop = overlay.topPt;
                overlay._startBottom = overlay.bottomPt;
                overlay._startMouseY = parent.y + mouse.y;
            }
            onPositionChanged: (mouse) => {
                var curY = parent.y + mouse.y;
                var dy = curY - overlay._startMouseY;
                var newBottom = overlay._startBottom + dy;

                // clamp: top + minHeight <= newBottom <= overlay.height
                newBottom = Math.max(overlay.topPt + overlay.minHeight, Math.min(newBottom, overlay.height));
                overlay.bottomPt = newBottom;
            }
        }
    }

    // Left handle (modifies `left` only; right is fixed)
    Rectangle {
        id: leftHandle
        width: overlay.handleSize
        height: overlay.handleSize * 3
        x: overlay.leftPt - width / 2
        y: (overlay.topPt + overlay.bottomPt) / 2 - height / 2
        color: "transparent"
        z: 3

        Image {
            anchors.centerIn: parent
            source: "../assets/icons/left_right_arrow.png"
            width: 21
            height: 21
        }

        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.SizeHorCursor
            onPressed: (mouse) => {
                overlay._startLeft = overlay.leftPt;
                overlay._startRight = overlay.rightPt;
                overlay._startMouseX = parent.x + mouse.x;
            }
            onPositionChanged: (mouse) => {
                var curX = parent.x + mouse.x;
                var dx = curX - overlay._startMouseX;
                var newLeft = overlay._startLeft + dx;

                // clamp: 0 <= newLeft <= right - minWidth
                newLeft = Math.max(0, Math.min(newLeft, overlay.rightPt - overlay.minWidth));
                overlay.leftPt = newLeft;
            }
        }
    }

    // Right handle (modifies `right` only; left is fixed)
    Rectangle {
        id: rightHandle
        width: overlay.handleSize;
        height: overlay.handleSize * 3
        x: overlay.rightPt - width / 2
        y: (overlay.topPt + overlay.bottomPt) / 2 - height / 2
        color: "transparent"
        z: 3

        /*Text {
            anchors.centerIn: parent; text: "↔"; color: "white"; font.pixelSize: 18
        }*/

        Image {
            anchors.centerIn: parent
            source: "../assets/icons/left_right_arrow.png"
            width: 21
            height: 21
        }

        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.SizeHorCursor
            onPressed: (mouse) => {
                overlay._startLeft = overlay.leftPt;
                overlay._startRight = overlay.rightPt;
                overlay._startMouseX = parent.x + mouse.x;
            }
            onPositionChanged: (mouse) => {
                var curX = parent.x + mouse.x;
                var dx = curX - overlay._startMouseX;
                var newRight = overlay._startRight + dx;

                // clamp: left + minWidth <= newRight <= overlay.width
                newRight = Math.max(overlay.leftPt + overlay.minWidth, Math.min(newRight, overlay.width));
                overlay.rightPt = newRight;
            }
        }
    }

    // optional: show a translucent mask outside cropRect (simple version)
    Rectangle {
        anchors.fill: parent
        color: "black"
        opacity: 0.35
        z: 1
        visible: true
        // use clipping to create a hole: draw on top and then the cropRect is above
    }
}