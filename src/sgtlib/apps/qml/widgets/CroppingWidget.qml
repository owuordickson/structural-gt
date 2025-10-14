import QtQuick
import QtQuick.Controls

Item {
    id: cropOverlay
    anchors.fill: parent
    visible: true

    // Expose to outside QMLs
    property alias cropArea: cropRect

    // rectangle coordinates (px, relative to overlay)
    property real leftPt: 0
    property real topPt: 0
    property real rightPt: width
    property real bottomPt: height

    // helpers / config
    property int handleSize: 24
    property int minWidth: 40
    property int minHeight: 40
    property color borderColor: "red"
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
            rightPt = 256;
            bottomPt = 256;
            _initialized = true;
        }
    }

    // visible crop rectangle (computed from coords)
    Rectangle {
        id: cropRect
        x: cropOverlay.leftPt
        y: cropOverlay.topPt
        width: Math.max(cropOverlay.minWidth, cropOverlay.rightPt - cropOverlay.leftPt)
        height: Math.max(cropOverlay.minHeight, cropOverlay.bottomPt - cropOverlay.topPt)
        color: "#00AEEF"
        opacity: 0.2
        border.width: 2
        border.color: cropOverlay.borderColor
        z: 2

        // move the whole rectangle by dragging inside
        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.SizeAllCursor
            onPressed: (mouse) => {
                // snapshot coords and mouse position (parent is cropRect here)
                cropOverlay._startLeft = cropOverlay.leftPt;
                cropOverlay._startRight = cropOverlay.rightPt;
                cropOverlay._startTop = cropOverlay.topPt;
                cropOverlay._startBottom = cropOverlay.bottomPt;
                cropOverlay._startMouseX = parent.x + mouse.x; // mouse pos relative to overlay
                cropOverlay._startMouseY = parent.y + mouse.y;
            }
            onPositionChanged: (mouse) => {
                var curX = parent.x + mouse.x;
                var curY = parent.y + mouse.y;
                var dx = curX - cropOverlay._startMouseX;
                var dy = curY - cropOverlay._startMouseY;

                // shift all coordinates while clamping to parent bounds
                var newLeft = cropOverlay._startLeft + dx;
                var newRight = cropOverlay._startRight + dx;
                var newTop = cropOverlay._startTop + dy;
                var newBottom = cropOverlay._startBottom + dy;

                // clamp horizontally
                var w = newRight - newLeft;
                if (newLeft < 0) {
                    newLeft = 0;
                    newRight = newLeft + w;
                }
                if (newRight > cropOverlay.width) {
                    newRight = cropOverlay.width;
                    newLeft = newRight - w;
                }
                // clamp vertically
                var h = newBottom - newTop;
                if (newTop < 0) {
                    newTop = 0;
                    newBottom = newTop + h;
                }
                if (newBottom > cropOverlay.height) {
                    newBottom = cropOverlay.height;
                    newTop = newBottom - h;
                }

                // apply
                cropOverlay.leftPt = newLeft;
                cropOverlay.rightPt = newRight;
                cropOverlay.topPt = newTop;
                cropOverlay.bottomPt = newBottom;
            }
        }
    }

    // helper function for clamp (optional; using Math.min/Math.max inline below)

    // --- Handles: top, bottom, left, right ---
    // Top handle (modifies `top` only; bottom is fixed)
    Rectangle {
        id: topHandle
        width: cropOverlay.handleSize * 3
        height: cropOverlay.handleSize
        x: (cropOverlay.leftPt + cropOverlay.rightPt) / 2 - width / 2
        y: cropOverlay.topPt - height / 2
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
                cropOverlay._startTop = cropOverlay.topPt;
                cropOverlay._startBottom = cropOverlay.bottomPt;
                cropOverlay._startMouseY = parent.y + mouse.y; // global relative to overlay
            }
            onPositionChanged: (mouse) => {
                var curY = parent.y + mouse.y;
                var dy = curY - cropOverlay._startMouseY;
                var newTop = cropOverlay._startTop + dy;

                // clamp: 0 <= newTop <= bottom - minHeight
                newTop = Math.max(0, Math.min(newTop, cropOverlay.bottomPt - cropOverlay.minHeight));
                cropOverlay.topPt = newTop;
            }
        }
    }

    // Bottom handle (modifies `bottom` only; top is fixed)
    Rectangle {
        id: bottomHandle
        width: cropOverlay.handleSize * 3
        height: cropOverlay.handleSize
        x: (cropOverlay.leftPt + cropOverlay.rightPt) / 2 - width / 2
        y: cropOverlay.bottomPt - height / 2
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
                cropOverlay._startTop = cropOverlay.topPt;
                cropOverlay._startBottom = cropOverlay.bottomPt;
                cropOverlay._startMouseY = parent.y + mouse.y;
            }
            onPositionChanged: (mouse) => {
                var curY = parent.y + mouse.y;
                var dy = curY - cropOverlay._startMouseY;
                var newBottom = cropOverlay._startBottom + dy;

                // clamp: top + minHeight <= newBottom <= cropOverlay.height
                newBottom = Math.max(cropOverlay.topPt + cropOverlay.minHeight, Math.min(newBottom, cropOverlay.height));
                cropOverlay.bottomPt = newBottom;
            }
        }
    }

    // Left handle (modifies `left` only; right is fixed)
    Rectangle {
        id: leftHandle
        width: cropOverlay.handleSize
        height: cropOverlay.handleSize * 3
        x: cropOverlay.leftPt - width / 2
        y: (cropOverlay.topPt + cropOverlay.bottomPt) / 2 - height / 2
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
                cropOverlay._startLeft = cropOverlay.leftPt;
                cropOverlay._startRight = cropOverlay.rightPt;
                cropOverlay._startMouseX = parent.x + mouse.x;
            }
            onPositionChanged: (mouse) => {
                var curX = parent.x + mouse.x;
                var dx = curX - cropOverlay._startMouseX;
                var newLeft = cropOverlay._startLeft + dx;

                // clamp: 0 <= newLeft <= right - minWidth
                newLeft = Math.max(0, Math.min(newLeft, cropOverlay.rightPt - cropOverlay.minWidth));
                cropOverlay.leftPt = newLeft;
            }
        }
    }

    // Right handle (modifies `right` only; left is fixed)
    Rectangle {
        id: rightHandle
        width: cropOverlay.handleSize;
        height: cropOverlay.handleSize * 3
        x: cropOverlay.rightPt - width / 2
        y: (cropOverlay.topPt + cropOverlay.bottomPt) / 2 - height / 2
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
                cropOverlay._startLeft = cropOverlay.leftPt;
                cropOverlay._startRight = cropOverlay.rightPt;
                cropOverlay._startMouseX = parent.x + mouse.x;
            }
            onPositionChanged: (mouse) => {
                var curX = parent.x + mouse.x;
                var dx = curX - cropOverlay._startMouseX;
                var newRight = cropOverlay._startRight + dx;

                // clamp: left + minWidth <= newRight <= cropOverlay.width
                newRight = Math.max(cropOverlay.leftPt + cropOverlay.minWidth, Math.min(newRight, cropOverlay.width));
                cropOverlay.rightPt = newRight;
            }
        }
    }


    Connections {
    target: cropRect

    function updateHandles() {
        // Hide handles temporarily if dimensions are invalid
        const valid = cropRect.width > 0 && cropRect.height > 0
        topHandle.visible = bottomHandle.visible = leftHandle.visible = rightHandle.visible = valid
        if (!valid)
            return

        // Position handles relative to current cropRect geometry
        topHandle.x = cropRect.x + cropRect.width / 2 - topHandle.width / 2
        topHandle.y = cropRect.y - topHandle.height / 2

        bottomHandle.x = cropRect.x + cropRect.width / 2 - bottomHandle.width / 2
        bottomHandle.y = cropRect.y + cropRect.height - bottomHandle.height / 2

        leftHandle.x = cropRect.x - leftHandle.width / 2
        leftHandle.y = cropRect.y + cropRect.height / 2 - leftHandle.height / 2

        rightHandle.x = cropRect.x + cropRect.width - rightHandle.width / 2
        rightHandle.y = cropRect.y + cropRect.height / 2 - rightHandle.height / 2
    }

    function onXChanged() { updateHandles() }
    function onYChanged() { updateHandles() }
    function onWidthChanged() { updateHandles() }
    function onHeightChanged() { updateHandles() }
}

}