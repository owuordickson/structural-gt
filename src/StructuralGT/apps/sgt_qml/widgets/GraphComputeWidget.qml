import QtQuick
import QtQuick.Controls
import QtQuick.Layouts


Item {
    id: graphComputeTbl
    Layout.preferredHeight: (numRows * tblRowHeight) + 5
    Layout.preferredWidth: parent.width
    Layout.leftMargin: 5
    Layout.rightMargin: 5

    property int numRows: graphComputeModel.rowCount()
    property int tblRowHeight: 25


    Connections {
        target: mainController

        function onImageChangedSignal(){
            numRows = graphComputeModel.rowCount();
        }

        function onTaskTerminatedSignal(success_val, msg_data){
            numRows = graphComputeModel.rowCount();
        }

    }

}