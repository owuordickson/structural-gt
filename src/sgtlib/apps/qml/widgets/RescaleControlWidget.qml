import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Basic as Basic
import Theme 1.0

Item {
    id: rescaleControl
    height: 180
    width: parent.width

    // Expose TextFields as properties to make them accessible
    property alias lblScale: lblScaling
    property int valueRole: Qt.UserRole + 4


    ColumnLayout {
        id: scalingContainer
        spacing: 5
        //Layout.alignment: Qt.AlignHCenter
        visible: imageController.enable_img_controls()

        Label {
            id: lblScaling
            text: "Re-scale to:"
            font.bold: true
            color: Theme.text
        }

        ListView {
            id: listViewScalingOptions
            width: 180
            height: 150
            model: imgScaleOptionModel

            ButtonGroup {
                id: btnGrpScales
                exclusive: true
            }

            delegate: Item {
                width: listViewScalingOptions.width
                height: 24

                RowLayout {
                    anchors.fill: parent

                    Basic.RadioButton {
                        id: rdoButton
                        text: model.text
                        ButtonGroup.group: btnGrpScales
                        property bool isChecked: model.value
                        checked: isChecked
                        onClicked: btnGrpScales.checkedButton = this
                        // Custom radio indicator
                        indicator: Rectangle {
                            width: 14
                            height: 14
                            radius: 7
                            y: (rdoButton.height - height) / 2   // center vertically
                            border.color: rdoButton.checked ? Theme.dodgerBlue : Theme.text
                            border.width: 2
                            color: "transparent"

                            Rectangle {
                                visible: rdoButton.checked
                                width: 8
                                height: 8
                                radius: 4
                                anchors.centerIn: parent
                                color: Theme.darkGray
                            }
                        }
                        contentItem: Label {
                            text: rdoButton.text
                            font: rdoButton.font
                            color: Theme.text
                            verticalAlignment: Text.AlignVCenter
                            leftPadding: rdoButton.indicator.width + 6
                        }
                        onCheckedChanged: {
                            if (isChecked !== checked) {  // Only update if there is a change
                                isChecked = checked
                                var val = checked ? 1 : 0;
                                var index = imgFilterModel.index(model.index, 0);
                                imgScaleOptionModel.setData(index, val, valueRole);
                                drpDownRescale.close();
                                imageController.apply_img_scaling();
                            }
                        }
                    }
                }
            }
        }


    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            scalingContainer.visible = imageController.enable_img_controls();
        }
    }

}
