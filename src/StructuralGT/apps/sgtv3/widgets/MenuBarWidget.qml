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
        MenuItem { text: "Add image"; onTriggered: console.log("add image clicked") }
        //MenuItem { text: "Add image folder"; onTriggered: console.log("add image folder clicked") }

        Menu { title: "Project..."
            MenuItem { text: "Create project"; onTriggered: console.log("create project clicked") }
            MenuItem { text: "Open project"; onTriggered: console.log("Open clicked") }
        }
        MenuSeparator{}

        MenuItem { text: "Save project as..."; enabled: false; onTriggered: console.log("save project") }
        MenuSeparator{}

        Menu { title: "Export graph as..."
            MenuItem { text: "Edge list"; enabled: false; onTriggered: console.log("edge list") }
            MenuItem { text: "Adjacency matix"; enabled: false; onTriggered: console.log("adj matrix") }
            MenuItem { text: "As gexf"; enabled: false; onTriggered: console.log("gexf") }
            MenuItem { text: "Save all"; enabled: false; onTriggered: console.log("save all") }
        }
        //MenuSeparator{}
    }
    Menu {
        title: "Tools"
        MenuItem { text: "Brightness/Contrast"; enabled: false; onTriggered: dialogBrightnessCtrl.open() }
        MenuItem { text: "Show Graph"; enabled: false; onTriggered: dialogShowGraph.open() }
    }
    Menu {
        title: "Filters"
        MenuItem { text: "Binary Filters"; enabled: false; onTriggered: dialogBinFilters.open() }

        MenuSeparator{}

        MenuItem { text: "Image Filters"; enabled: false; onTriggered: dialogImgFilters.open() }
    }
    Menu {
        title: "Analyze"
        //MenuItem { text: "GT Metrics"; onTriggered: console.log("GT clicked") }
        Menu { title: "Graph Metrics"
            MenuItem { text: "Current Image"; enabled: true; onTriggered: dialogGTOptions.open() }
            MenuItem { text: "All Images"; enabled: false; onTriggered: dialogGTOptions.open() }
        }
    }
    Menu {
        title: "Help"
        MenuItem { text: "Structural GT Help"; onTriggered: dialogAbout.open() }
    }
}

