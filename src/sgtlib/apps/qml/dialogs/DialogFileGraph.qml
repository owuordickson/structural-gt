import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs as QuickDialogs

QuickDialogs.FileDialog {
    id: graphFileDialog
    title: "Open file"
    nameFilters: [projectController.get_file_extensions("graph")]
    onAccepted: {
        projectController.upload_graph_file(graphFileDialog.selectedFile);
    }
    //onRejected: console.log("File selection canceled")
}