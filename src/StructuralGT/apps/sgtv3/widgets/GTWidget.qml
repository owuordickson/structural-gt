import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Repeater {

    model: [
        {id: "display_heatmaps", text: "Display Heatmaps"},
        {id: "display_degree_histogram", text: "Average Degree"},
        {id: "compute_network_diameter", text: "Network Diameter"},
        {id: "compute_graph_density", text: "Graph Density"},
        {id: "compute_wiener_index", text: "Wiener Index"},
        {id: "compute_node_connectivity", text: "Average Node Connectivity"},
        {id: "compute_global_efficiency", text: "Global Coefficient"},
        {id: "compute_clustering_coef", text: "Average Clustering Coefficient"},
        {id: "compute_assortativity_coef", text: "Assortativity Coefficient"},
        {id: "display_betweenness_histogram", text: "Betweenness Centrality"},
        {id: "display_closeness_histogram", text: "Closenness Centrality"},
        {id: "display_eigenvector_histogram", text: "Eigenvector Centrality"},
        {id: "display_ohms_histogram", text: "Ohms Centrality"},
        {id: "display_percolation_histogram", text: "Percolation Centrality"}
    ]

    delegate: RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 10
        Layout.alignment: Qt.AlignLeft

        CheckBox {
            id: checkBox
            objectName: modelData.id
            //Layout.preferredWidth: 100
            text: modelData.text
            checked: mainController.load_gte_setting(modelData.id)
        }
    }

}
