import QtQuick
import QtQuick.Controls
//import Qt.labs.platform


MenuBar {
    property int valueRole: Qt.UserRole + 4

    Menu {
        title: projectController.get_sgt_title()
        MenuItem { text: "&About"; onTriggered: dialogAbout.open(); }
        MenuSeparator{}
        MenuItem { text: "&Quit"; onTriggered: Qt.quit(); }
    }
    Menu {
        title: "File"
        MenuItem { text: "Add image"; onTriggered: dialogFileImage.open() }
        MenuItem { text: "Add image folder"; onTriggered: dialogFolderImage.open() }
        Menu {
            id: mnuImportGraphFrom
            title: "Import graph from..."
            MenuItem {id: mnuImportCSV; text:"CSV (adj. matrix, edge list, xyz)"; enabled: true; onTriggered: dialogFileGraph.open()}
            MenuItem {id: mnuImportGSD; text:"GSD/HOOMD"; enabled: true; onTriggered: dialogFileGraph.open()}
        }
        MenuSeparator{}

        Menu { title: "Project..."
            MenuItem { text: "Create project"; onTriggered: dialogCreateProject.open() }
            MenuItem { text: "Open project"; onTriggered: dialogFileProject.open() }
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
        MenuItem {id:mnuImgFilter; text: "Image Filters"; enabled: false; onTriggered: dialogImgFilters.open() }
        MenuItem {id: mnuImgColors; text: "Image Colors"; enabled: false; onTriggered: imgColorsWindow.visible = true}
        MenuItem {id: mnuImgHistogram; text: "Calculate Histogram"; enabled: false; onTriggered: imgHistogramWindow.visible = true}
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
        graphController.export_graph_to_file();
    }
    
    
    function save_processed_images (row) {

        for (let i = 0; i < saveImgModel.rowCount(); i++) {
            let val = i === row ? 1 : 0;
            var index = saveImgModel.index(i, 0);
            saveImgModel.setData(index, val, valueRole);
        }
        imageController.save_img_files();
    }
    

    function save_project () {

        let is_open = projectController.is_project_open();
        if (is_open === false) {
            dialogAlert.title = "Save Error";
            dialogAlert.alertMsg = "Please create/open the SGT project first, then try again.";
            dialogAlert.alertColor = Theme.blueText;
            dialogAlert.open();
        } else {
            projectController.run_save_project();
        }

    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            mnuSaveProjAs.enabled = imageController.display_image();
            mnuExportEdges.enabled = graphPropsModel.rowCount() > 0;
            mnuExportNodes.enabled = graphPropsModel.rowCount() > 0;
            mnuExportAdj.enabled = graphPropsModel.rowCount() > 0;
            mnuExportGexf.enabled = graphPropsModel.rowCount() > 0;
            mnuExportGSD.enabled = graphPropsModel.rowCount() > 0;
            mnuExportAll.enabled = graphPropsModel.rowCount() > 0;

            //mnuRescaleImgCtrl.enabled = imageController.display_image();  HAS ERRORS
            mnuBrightnessImgCtrl.enabled = imageController.display_image();
            mnuContrastImgCtrl.enabled = imageController.display_image();
            mnuBinImgFilter.enabled = imageController.display_image();
            mnuImgFilter.enabled = imageController.display_image();
            mnuSoloAnalze.enabled = imageController.display_image();
            mnuMultiAnalyze.enabled = imageController.display_image();
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


    Connections {
        target: imageController

        function onShowImageFilterControls(allow) {
            if (allow) {
                mnuImgHistogram.enabled = imageController.enable_img_controls();
                mnuImgColors.enabled = imageController.enable_img_controls();
            } else {
                mnuImgHistogram.enabled = allow;
                mnuImgColors.enabled = allow;
            }
        }
    }
}

