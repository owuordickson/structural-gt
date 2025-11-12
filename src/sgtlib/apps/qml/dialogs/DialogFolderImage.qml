import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt.labs.platform as Platform

Platform.FolderDialog {
    id: imageFolderDialog
    title: "Select a Folder"
    onAccepted: {
        //console.log("Selected folder:", folder)
        projectController.upload_multiple_images(imageFolderDialog.folder);
    }
    //onRejected: {console.log("Canceled")}
}