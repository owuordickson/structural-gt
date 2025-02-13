import QtQuick
import QtQuick.Controls
import QtQuick.Layouts



ColumnLayout {
    Layout.fillWidth: true
    Layout.fillHeight: true
    Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
    //anchors.fill: parent
    //spacing: 10

    Label {
        id: imgLabel
        text: "No image loaded"
        visible: !imageController.is_image_loaded()
    }

    Image {
        id: imgView
        width: parent.width
        height: parent.height
        //width: 300
        //height: 300
        Layout.fillWidth: true
        Layout.fillHeight: true
        fillMode: Image.PreserveAspectFit
        source: ""
        visible: imageController.is_image_loaded()
    }

    /*Connections: {
            target: imageProvider
            //imgView.source = imageProvider.getPixmap()
            onImage_changed: {
                imgView.source = imageProvider.get_pixmap()
            }
        }*/

    Component.onCompleted: {
        imgView.source = imageController.get_pixmap()
        //imgView.source = "image://imageProvider"
        //imgView.source = "image://imageProvider/graph_icon"
    }
}


