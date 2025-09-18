import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: starRating
    width: 384
    height: 48
    Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

    property real rating: 0       // current rating (0-10, can be halves)
    property int maxStars: 10

    Label {
        Layout.alignment: Qt.AlignHCenter
        text: "How good is the graph? Pick a score: 0 - 10"
        color: "#2266ff"
    }

    Row {
        id: starsRow
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
                                starRating.rating = index + 0.5
                            else
                                starRating.rating = index + 1
                        }
                    }
                }

                Label {
                    Layout.alignment: Qt.AlignHCenter
                    text: index + 1
                    font.pixelSize: 9
                }
            }
        }
    }
}