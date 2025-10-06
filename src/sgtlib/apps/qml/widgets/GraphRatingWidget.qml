import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: ratingContainer
    width: 384
    height: 64
    Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

    property real rating: 0       // current rating (0-10, can be halves)
    property int maxStars: 10

    Label {
        Layout.alignment: Qt.AlignHCenter
        text: "How good is the graph? Pick a score: 0 - 9."
        color: "#2266ff"
    }

    Row {
        id: starsRow
        Layout.alignment: Qt.AlignHCenter
        spacing: 4

        Repeater {
            model: maxStars
            delegate: ColumnLayout {

                Item {
                    id: starItem
                    width: 28
                    height: 28

                    Image {
                        id: starImg
                        anchors.fill: parent
                        source: (rating >= index + 1) ? "../assets/icons/star-full.png" : (rating >= index + 0.5) ? "../assets/icons/star-half.png" : "../assets/icons/star-none.png"
                    }

                    MouseArea {
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: (mouse) => {
                            // Determine if clicked left or right half for half/full star
                            let localX = mouse.x
                            if (localX < starItem.width / 2)
                                ratingContainer.rating = index + 0.5
                            else
                                ratingContainer.rating = index + 1
                        }
                    }
                }

                Label {
                    Layout.alignment: Qt.AlignHCenter
                    text: index
                    font.pixelSize: 10
                }
            }
        }
    }

    Label {
        Layout.fillWidth: true
        Layout.topMargin: 15
        leftPadding: 10
        rightPadding: 10
        horizontalAlignment: Text.AlignJustify  // Justify the text
        //Layout.alignment: Qt.AlignHCenter
        text: "Disclaimer: By rating the graph, you agree to share your rating and the graph image with the StructuralGT developers, and this information may be used to train AI models."
        color: "#909090"
        font.pixelSize: 8
        font.bold: true
        wrapMode: Text.Wrap
    }
}