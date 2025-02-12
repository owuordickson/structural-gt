import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Repeater {

    model: [
        {id: "cbxDispHeatmaps", text: "Display Heatmaps"},
        {id: "cbxAvgDegree", text: "Average Degree"},
        {id: "cbxGraphDensity", text: "Network Diameter"},
        {id: "cbx", text: "Graph Density"},
        {id: "cbxWienerIndex", text: "Wiener Index"},
        {id: "cbxAvgNodeConn", text: "Average Node Connectivity"},
        {id: "cbxzGlobalCoeff", text: "Global Coefficient"},
        {id: "cbxAvgClustering", text: "Average Clustering Coefficient"},
        {id: "cbxAssortativity", text: "Assortativity Coefficient"},
        {id: "cbxBtwnCentrality", text: "Betweenness Centrality"},
        {id: "cbxClseCentrality", text: "Closenness Centrality"},
        {id: "cbxEigenvector", text: "Eigenvector Centrality"},
        {id: "cbxOhmsCentrality", text: "Ohms Centrality"},
        {id: "cbxPercolation", text: "Percolation Centrality"}
    ]

    delegate: RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 10
        Layout.alignment: Qt.AlignLeft

        CheckBox {
            id: checkBox
            //Layout.preferredWidth: 100
            text: modelData.text
            checked: false
        }
    }

}
