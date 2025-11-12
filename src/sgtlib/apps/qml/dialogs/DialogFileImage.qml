import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs as QuickDialogs

QuickDialogs.FileDialog {
    id: imageFileDialog
    title: "Open image file"
    nameFilters: [projectController.get_file_extensions("img")]
    onAccepted: {
        projectController.upload_single_image(imageFileDialog.selectedFile);
    }
    //onRejected: console.log("File selection canceled")
}