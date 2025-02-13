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
        Menu { title: "Project..."
            MenuItem { text: "Create project"; onTriggered: console.log("create project clicked") }
            MenuItem { text: "Open project"; onTriggered: console.log("Open clicked") }
        }
        MenuSeparator{}

        MenuItem { text: "Save"; onTriggered: console.log("save project") }
        MenuSeparator{}

        Menu { title: "Export graph..."
            MenuItem { text: "Edge list"; onTriggered: console.log("edge list") }
            MenuItem { text: "Adjacency matix"; onTriggered: console.log("adj matrix") }
            MenuItem { text: "As gexf"; onTriggered: console.log("gexf") }
            MenuItem { text: "Save all"; onTriggered: console.log("save all") }
        }
        //MenuSeparator{}
    }
    Menu {
        title: "Tools"
        MenuItem { text: "Brightness/Contrast"; onTriggered: dialogBrightnessCtrl.open() }
        MenuItem { text: "Show Graph"; onTriggered: dialogShowGraph.open() }
    }
    Menu {
        title: "Filters"
        MenuItem { text: "Binary Filters";  onTriggered: dialogBinFilters.open() }

        MenuSeparator{}

        MenuItem { text: "Image Filters"; onTriggered: dialogImgFilters.open() }
    }
    Menu {
        title: "Analyze"
        //MenuItem { text: "GT Metrics"; onTriggered: console.log("GT clicked") }
        Menu { title: "Graph Metrics"
            MenuItem { text: "Current Image"; onTriggered: dialogGTOptions.open() }
            MenuItem { text: "All Images"; onTriggered: dialogGTOptions.open() }
        }
    }
    Menu {
        title: "Help"
        MenuItem { text: "Structural GT Help"; onTriggered: dialogAbout.open() }
    }
}

