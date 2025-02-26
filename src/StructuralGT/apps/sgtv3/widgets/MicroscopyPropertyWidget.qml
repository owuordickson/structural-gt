import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: microscopyProps  // used for external access
    //height: parent.height
    width: 300
    //Layout.fillHeight: true
    //Layout.fillWidth: true
    enabled: mainController.display_image();

    property int txtWidthSize: 170
    property int lblWidthSize: 100
    property int valueRole: Qt.UserRole + 4

    ColumnLayout {
        id: microscopyPropsLayout
        //spacing: 10

        Repeater {
            model: microscopyPropsModel
            delegate: RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 10
                Layout.alignment: Qt.AlignLeft

                Label {
                    id: label
                    wrapMode: Text.Wrap
                    Layout.preferredWidth: lblWidthSize
                    text: model.text
                }

                TextField {
                    id: txtField
                    objectName: model.id
                    Layout.fillWidth: true
                    Layout.minimumWidth: txtWidthSize
                    Layout.rightMargin: 10
                    text: model.value
                    onEditingFinished: {
                        var index = microscopyPropsModel.index(model.index, 0);
                        microscopyPropsModel.setData(index, text, valueRole);
                        mainController.apply_microscopy_props_changes();
                    }
                }
            }
        }

    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            microscopyProps.enabled = mainController.display_image();
        }

    }

}
