import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt.labs.platform as Platform

Platform.FolderDialog {
    id: outFolderDialog
    title: "Select a Folder"
    onAccepted: {
        //console.log("Selected folder:", folder)
        projectController.set_output_dir(folder)
    }
    //onRejected: {console.log("Canceled")}
}