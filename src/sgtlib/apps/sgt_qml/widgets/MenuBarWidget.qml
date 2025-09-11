import QtQuick
import QtQuick.Controls
//import Qt.labs.platform


MenuBar {
    property int valueRole: Qt.UserRole + 4

    Menu {
        title: mainController.get_sgt_title()
        MenuItem { text: "&About"; onTriggered: dialogAbout.open(); }
        MenuSeparator{}
        MenuItem { text: "&Quit"; onTriggered: Qt.quit(); }
    }
    Menu {
        title: "File"
        MenuItem { text: "Add image"; onTriggered: imageFileDialog.open() }
        MenuItem { text: "Add image folder"; onTriggered: imageFolderDialog.open() }
        Menu {
            id: mnuImportGraphFrom
            title: "Import graph from..."
            MenuItem {id: mnuImportCSV; text:"CSV (adj. matrix, edge list, xyz)"; enabled: true; onTriggered: graphFileDialog.open()}
            MenuItem {id: mnuImportGSD; text:"GSD/HOOMD"; enabled: true; onTriggered: graphFileDialog.open()}
        }
        MenuSeparator{}

        Menu { title: "Project..."
            MenuItem { text: "Create project"; onTriggered: createProjectDialog.open() }
            MenuItem { text: "Open project"; onTriggered: projectFileDialog.open() }
        }

        MenuItem {id: mnuSaveProjAs; text: "Save project"; enabled: false; onTriggered: save_project() }
        MenuSeparator{}

        MenuItem {id:mnuExportAll; text: "Save images"; enabled: false; onTriggered: save_processed_images(0) }
        MenuSeparator{}

        Menu {
            id: mnuExportGraphAs
            title: "Export graph as..."
            MenuItem {id:mnuExportEdges; text: "Edge list"; enabled: false; onTriggered: export_graph_data(0) }
            MenuItem {id:mnuExportNodes; text: "Node positions"; enabled: false; onTriggered: export_graph_data(1) }
            MenuItem {id:mnuExportAdj; text: "Adjacency matrix"; enabled: false; onTriggered: export_graph_data(3) }
            MenuItem {id:mnuExportGexf; text: "As gexf"; enabled: false; onTriggered: export_graph_data(2) }
            MenuItem {id:mnuExportGSD; text: "As GSD/HOOMD"; enabled: false; onTriggered: export_graph_data(4) }
        }

    }
    Menu {
        id: mnuImgCtrls
        title: "Tools"
        //MenuItem {id:mnuRescaleImgCtrl; text: "Rescale Image"; enabled: false; onTriggered: dialogRescaleCtrl.open() }
        MenuItem {id:mnuBrightnessImgCtrl; text: "Brightness/Contrast"; enabled: false; onTriggered: dialogBrightnessCtrl.open() }
        MenuItem {id:mnuContrastImgCtrl; text: "Show Graph"; enabled: false; onTriggered: dialogExtractGraph.open() }
    }
    Menu {
        id: mnuImgFilters
        title: "Filters"
        MenuItem {id:mnuBinImgFilter; text: "Binary Filters"; enabled: false; onTriggered: dialogBinFilters.open() }

        MenuSeparator{}

        MenuItem {id:mnuImgFilter; text: "Image Filters"; enabled: false; onTriggered: dialogImgFilters.open() }
    }
    Menu {
        id: mnuAnalyze
        title: "Analyze"
        Menu { title: "GT Parameters"
            MenuItem {id:mnuSoloAnalze; text: "Current Image"; enabled: false; onTriggered: dialogRunAnalyzer.open() }
            MenuItem {id:mnuMultiAnalyze; text: "All Images"; enabled: false; onTriggered: dialogRunMultiAnalyzer.open() }
        }
    }
    Menu {
        title: "Help"
        MenuItem { id:mnuHelp; text: "StructuralGT Help"; enabled: true; onTriggered: dialogAbout.open() }
        MenuItem { id:mnuLogs; text: "View Logs"; enabled: true; onTriggered: loggingWindowPanel.visible = true }
    }

    function export_graph_data (row) {

        for (let i = 0; i < exportGraphModel.rowCount(); i++) {
            let val = i === row ? 1 : 0;
            var index = exportGraphModel.index(i, 0);
            exportGraphModel.setData(index, val, valueRole);
        }
        mainController.export_graph_to_file();
    }
    
    
    function save_processed_images (row) {

        for (let i = 0; i < saveImgModel.rowCount(); i++) {
            let val = i === row ? 1 : 0;
            var index = saveImgModel.index(i, 0);
            saveImgModel.setData(index, val, valueRole);
        }
        mainController.save_img_files();
    }
    

    function save_project () {

        let is_open = mainController.is_project_open();
        if (is_open === false) {
            dialogAlert.title = "Save Error";
            lblAlertMsg.text = "Please create/open the SGT project first, then try again.";
            lblAlertMsg.color = "#2255bc";
            dialogAlert.open();
        } else {
            let success_val = mainController.run_save_project();
        }

    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            mnuSaveProjAs.enabled = mainController.display_image();
            mnuExportEdges.enabled = graphPropsModel.rowCount() > 0;
            mnuExportNodes.enabled = graphPropsModel.rowCount() > 0;
            mnuExportAdj.enabled = graphPropsModel.rowCount() > 0;
            mnuExportGexf.enabled = graphPropsModel.rowCount() > 0;
            mnuExportGSD.enabled = graphPropsModel.rowCount() > 0;
            mnuExportAll.enabled = graphPropsModel.rowCount() > 0;

            //mnuRescaleImgCtrl.enabled = mainController.display_image();  HAS ERRORS
            mnuBrightnessImgCtrl.enabled = mainController.display_image();
            mnuContrastImgCtrl.enabled = mainController.display_image();
            mnuBinImgFilter.enabled = mainController.display_image();
            mnuImgFilter.enabled = mainController.display_image();
            mnuSoloAnalze.enabled = mainController.display_image();
            mnuMultiAnalyze.enabled = mainController.display_image();
        }

        function onTaskTerminatedSignal(success_val, msg_data) {
            mnuExportEdges.enabled = graphPropsModel.rowCount() > 0;
            mnuExportNodes.enabled = graphPropsModel.rowCount() > 0;
            mnuExportAdj.enabled = graphPropsModel.rowCount() > 0;
            mnuExportGexf.enabled = graphPropsModel.rowCount() > 0;
            mnuExportGSD.enabled = graphPropsModel.rowCount() > 0;
            mnuExportAll.enabled = graphPropsModel.rowCount() > 0;
        }
    }
}

