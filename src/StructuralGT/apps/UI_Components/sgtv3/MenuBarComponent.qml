import QtQuick
import QtQuick.Controls

MenuBar {
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
            MenuSeparator{}

            MenuItem { text: "Quit"; onTriggered: console.log("quit") }
        }
        Menu {
            title: "Tools"
            MenuItem { text: "Crop"; onTriggered: console.log("") }
            MenuItem { text: "Brightness/Contrast"; onTriggered: console.log("brightness/contrast") }
        }
        Menu {
            title: "Filters"
            Menu { title: "Binary Filters"
                Menu { title: "Threshold Type"
                    MenuItem { text: "Adaptive"; onTriggered: console.log("adaptive") }
                    MenuItem { text: "Global"; onTriggered: console.log("global") }
                    MenuItem { text: "OTSU"; onTriggered: console.log("otsu") }
                }
                MenuItem { text: "Apply Dark Background"; onTriggered: console.log("dark bg") }
            }
            MenuSeparator{}

            Menu { title: "Image Filters"
                MenuItem { text: "Auto Level"; onTriggered: console.log("auto level") }
                MenuItem { text: "Gaussian Blur"; onTriggered: console.log("gaussian") }
                MenuItem { text: "Laplacian Blur"; onTriggered: console.log("laplacian") }
                MenuItem { text: "Lowpass Filter"; onTriggered: console.log("lowpass") }
                MenuItem { text: "LUT Gamma"; onTriggered: console.log("gamma") }
                MenuItem { text: "Median"; onTriggered: console.log("median") }
                MenuItem { text: "Scharr"; onTriggered: console.log("scharr") }
                MenuItem { text: "Sobel"; onTriggered: console.log("sobel") }
            }
        }
        Menu {
            title: "Analyze"
            MenuItem { text: "GT Metrics"; onTriggered: console.log("GT clicked") }
        }
        Menu {
            title: "Help"
            MenuItem { text: "Tutorials"; onTriggered: console.log("Tutorials clicked") }
            MenuItem { text: "About"; onTriggered: console.log("About clicked") }
        }
    }
