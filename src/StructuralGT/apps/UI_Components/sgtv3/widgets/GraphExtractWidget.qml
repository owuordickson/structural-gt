import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Repeater {
    id: extractRepeater
    model: [
        { id: "cbxWeight", text: "Add Weights", subitems: [
                { id: "cbxByDiameter", text: "by diameter" },
                { id: "cbxByArea", text: "by area" },
                { id: "cbxByLength", text: "by length" },
                { id: "cbxByAngle", text: "by angle" },
                { id: "cbxByInvLength", text: "by inverse-length" },
                { id: "cbxByConductance", text: "by conductance" },
                { id: "cbxByResistance", text: "by resistance" }
            ] },
        { id: "cbxMerge", text: "Merge Nearby Nodes" },
        { id: "cbxPrune", text: "Prune Dangling Edges" },
        { id: "cbxRmDisconn", text: "Remove Disconnected Segments" },
        { id: "cbxRmLoops", text: "Remove Self Loops" },
        { id: "cbxMultigraph", text: "Is Multigraph?" },
        { id: "cbxDisplayID", text: "Display Node ID" }
    ]

    delegate: ColumnLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 10

        RowLayout {
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignLeft

            CheckBox {
                id: checkBox
                text: modelData.text
                checked: false
            }
        }

        /*Loader {
            id: controlLoader
            active: checkBox.checked && extractRepeater.modelData.id === "cbxWeight"
            sourceComponent: wgtComponent
        }*/

        /*Loader {
            id: controlLoader
            sourceComponent: extractRepeater.modelData.id === "cbxWeight" ? wgtComponent : ""
        }*/
    }

    /*Component {
        id: wgtComponent

        ColumnLayout {
            Layout.fillWidth: true

            Repeater {
                id: wgtRepeater
                model: [
                    { id: "cbxByDiameter", text: "by diameter" },
                    { id: "cbxByArea", text: "by area" },
                    { id: "cbxByLength", text: "by length" },
                    { id: "cbxByAngle", text: "by angle" },
                    { id: "cbxByInvLength", text: "by inverse-length" },
                    { id: "cbxByConductance", text: "by conductance" },
                    { id: "cbxByResistance", text: "by resistance" }
                ]

                delegate: RowLayout {
                    Layout.fillWidth: true
                    Layout.leftMargin: 20
                    Layout.alignment: Qt.AlignLeft

                    CheckBox {
                        text: wgtRepeater.modelData.text
                        checked: false
                    }
                }
            }
        }
    }*/

    /*Component {
        id: wgtComponent

        ColumnLayout {
            Layout.fillWidth: true

            ListView {
                id: wgtListView
                model: [
                    { id: "cbxByDiameter", text: "by diameter" },
                    { id: "cbxByArea", text: "by area" },
                    { id: "cbxByLength", text: "by length" },
                    { id: "cbxByAngle", text: "by angle" },
                    { id: "cbxByInvLength", text: "by inverse-length" },
                    { id: "cbxByConductance", text: "by conductance" },
                    { id: "cbxByResistance", text: "by resistance" }
                ]

                height: wgtListView.count > 0 ? wgtListView.contentHeight : 0
                clip: true
                interactive: false

                delegate: RowLayout {
                    Layout.fillWidth: true
                    Layout.leftMargin: 20
                    Layout.alignment: Qt.AlignLeft

                    CheckBox {
                        text: modelData.text
                        checked: false
                    }
                }
            }
        }
    }*/
}
