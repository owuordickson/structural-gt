import QtQuick
import QtQuick.Controls
//import Qt.labs.platform


MenuBar {
    Menu {
        title: "Structural GT"
        MenuItem { text: "&About"; onTriggered: dialogAbout.open(); }
        MenuSeparator{}
        MenuItem { text: "&Quit"; onTriggered: Qt.quit(); }
    }

    Menu {
        title: "File"
        MenuItem { text: "Add image"; onTriggered: imageFileDialog.open() }
        //MenuItem { text: "Add image folder"; onTriggered: console.log("add image folder clicked") }

        Menu { title: "Project..."
            MenuItem { text: "Create project"; onTriggered: createProjectDialog.open() }
            MenuItem { text: "Open project"; onTriggered: projectFileDialog.open() }
        }
        MenuSeparator{}

        MenuItem {id: mnuSaveProjAs; text: "Save project as..."; enabled: mainController.display_image(); onTriggered: saveProjectDialog.open() }
        MenuSeparator{}

        Menu {
            id: mnuExportGraphAs
            title: "Export GT graph as..."
            enabled: true
            MenuItem {id:mnuExportEdge; text: "Edge list"; enabled: false; onTriggered: console.log("edge list") }
            MenuItem {id:mnuExportAdj; text: "Adjacency matix"; enabled: false; onTriggered: console.log("adj matrix") }
            MenuItem {id:mnuExportGexf; text: "As gexf"; enabled: false; onTriggered: console.log("gexf") }
            MenuItem {id:mnuExportAll; text: "Save all"; enabled: false; onTriggered: console.log("save all") }
        }
        //MenuSeparator{}
    }
    Menu {
        id: mnuImgCtrls
        title: "Tools"
        enabled: true
        MenuItem {id:mnuBrightnessImgCtrl; text: "Brightness/Contrast"; enabled: false; onTriggered: dialogBrightnessCtrl.open() }
        MenuItem {id:mnuContrastImgCtrl; text: "Show Graph"; enabled: false; onTriggered: dialogExtractGraph.open() }
    }
    Menu {
        id: mnuImgFilters
        title: "Filters"
        enabled: true
        MenuItem {id:mnuBinImgFilter; text: "Binary Filters"; enabled: false; onTriggered: dialogBinFilters.open() }

        MenuSeparator{}

        MenuItem {id:mnuImgFilter; text: "Image Filters"; enabled: false; onTriggered: dialogImgFilters.open() }
    }
    Menu {
        id: mnuAnalyze
        title: "Analyze"
        enabled: true
        Menu { title: "GT Parameters"
            MenuItem {id:mnuSoloAnalze; text: "Current Image"; enabled: false; onTriggered: dialogRunAnalyzer.open() }
            MenuItem {id:mnuMultiAnalyze; text: "All Images"; enabled: false; onTriggered: dialogRunMultiAnalyzer.open() }
        }
    }
    Menu {
        title: "Help"
        MenuItem { id:mnuHelp; text: "Structural GT Help"; enabled: true; onTriggered: dialogAbout.open() }
    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            mnuSaveProjAs.enabled = mainController.display_image();

            mnuExportEdge.enabled = graphPropsTableModel.rowCount() > 0 ? true : false;
            mnuExportAdj.enabled = graphPropsTableModel.rowCount() > 0 ? true : false;
            mnuExportGexf.enabled = graphPropsTableModel.rowCount() > 0 ? true : false;
            mnuExportAll.enabled = graphPropsTableModel.rowCount() > 0 ? true : false;

            mnuBrightnessImgCtrl.enabled = mainController.display_image();
            mnuContrastImgCtrl.enabled = mainController.display_image();
            mnuBinImgFilter.enabled = mainController.display_image();
            mnuImgFilter.enabled = mainController.display_image();
            mnuSoloAnalze.enabled = mainController.display_image();
            mnuMultiAnalyze.enabled = mainController.display_image();
        }
    }
}

