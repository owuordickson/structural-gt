import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs as QuickDialogs

QuickDialogs.FileDialog {
    id: projectFileDialog
    title: "Open .sgtproj file"
    nameFilters: [projectController.get_file_extensions("proj")]
    onAccepted: {
        projectController.open_sgt_project(projectFileDialog.selectedFile);
    }
    //onRejected: console.log("File selection canceled")
}