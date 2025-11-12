import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "widgets"
import "layouts"
import "windows"
import "dialogs"

ApplicationWindow {
    id: mainWindow
    width: 1024
    height: 824
    visible: true
    title: projectController.get_sgt_title();
    font.family: "Arial"  // or Qt.application.font.family
    color: "#f0f0f0"

    // Top Menu Bar
    menuBar: MenuBar {
    }

    // Bottom Footer Bar
    footer: StatusBarLayout {
    }

    // Main Layout
    GridLayout {
        anchors.fill: parent
        rows: 2
        columns: 2

        // First row, first column (spanning 2 columns) - Ribbon
        Rectangle {
            Layout.row: 0
            Layout.column: 0
            Layout.columnSpan: 2
            Layout.topMargin: 5
            Layout.leftMargin: 10
            Layout.rightMargin: 10
            Layout.alignment: Qt.AlignTop
            Layout.preferredHeight: 40
            Layout.preferredWidth: parent.width
            color: "#f0f0f0"
            RibbonLayout {
            }
        }

        // Second row, first column - Left Navigation Pane
        Rectangle {
            id: recLeftPane
            Layout.row: 1
            Layout.column: 0
            Layout.leftMargin: 10
            Layout.rightMargin: 5
            Layout.preferredHeight: parent.height - 40
            Layout.preferredWidth: 300
            color: "#f0f0f0"
            LeftLayout {
            }
        }

        // Second row, second column - Center Content
        Rectangle {
            id: recCenterContent
            Layout.row: 1
            Layout.column: 1
            Layout.rightMargin: 10
            Layout.preferredHeight: parent.height - 40
            Layout.preferredWidth: parent.width - 300
            Layout.fillWidth: true
            color: "#f0f0f0"
            MainLayout {
            }
        }

        // Logging Window
        LoggingWindow {
            id: loggingWindowPanel
        }

        // Histogram Window
        ImageHistogramWindow
        {
            id: imgHistogramWindow
        }

        // Image Colors Window
        ImageColorsWindow
        {
            id: imgColorsWindow
        }

        // Image Cropping Window
        ImageCroppingWindow {
            id: imgCroppingWindow
        }
    }

    // Full Page Progress Spinner
    LoadingSpinnerOverlay{}

    // Left Pane Toggle Button
    function toggleLeftPane(showVal) {
        recLeftPane.visible = showVal;
    }

    // Dialogs
    DialogAbout{id: dialogAbout}
    DialogAlert{id: dialogAlert}
    DialogCreateProject{id: dialogCreateProject}
    DialogBrightnessControl{id: dialogBrightnessCtrl}
    DialogRescaleControl{id: dialogRescaleCtrl}
    DialogExtractGraph{id: dialogExtractGraph}
    DialogBinaryFilters{id: dialogBinFilters}
    DialogImageFilters{id: dialogImgFilters}
    DialogGraphAnalyzer{id: dialogRunAnalyzer}
    DialogGraphMultiAnalyzer{id: dialogRunMultiAnalyzer}
    DialogFolderOutput{id: dialogFolderOutput}
    DialogFolderImage{id: dialogFolderImage}
    DialogFileImage{id: dialogFileImage}
    DialogFileGraph{id: dialogFileGraph}
    DialogFileProject{id: dialogFileProject}

}
