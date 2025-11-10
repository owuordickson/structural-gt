import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts


ColumnLayout {
    Layout.fillWidth: true
    Layout.fillHeight: true
    Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

    // The Welcome View
    WelcomeWidget{}

    // Image View Controls
    ImageViewControls{}

    // Image Container
    ImageWidget{}

    // Image Navigation Controls
    ImageNavControls{}

}


