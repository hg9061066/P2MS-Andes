import torch
import numpy as np
import folium
import os
from src.model import P2MSAndesNet

def generate_risk_map():
    print("--- Generating Geographical Risk Dashboard ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Privacy-Hardened Global Model
    model = P2MSAndesNet()
    model.load_state_dict(torch.load("data/processed/global_model.pth"))
    model.to(device).eval()

    # 2. Define Node Coordinates & Metadata
    # In a real system, these would come from your GIS database
    nodes_metadata = {
        "Node_1_Neuquen": {"coords": [-38.95, -68.06], "name": "Neuquén, AR"},
        "Node_2_Rio_Negro": {"coords": [-41.13, -71.30], "name": "Río Negro, AR"},
        "Node_3_Valparaiso": {"coords": [-33.04, -71.61], "name": "Valparaíso, CL"},
        "Node_4_Aysen": {"coords": [-45.57, -72.06], "name": "Aysén, CL"}
    }

    # Create the base map centered over the Andes
    m = folium.Map(location=[-40.0, -70.0], zoom_start=5, tiles="cartodbpositron")

    colors = {0: "green", 1: "orange", 2: "red"}
    labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk (Alert)"}

    # 3. Perform Inference for each Node
    for node_id, info in nodes_metadata.items():
        file_path = f"data/processed/{node_id}.npz"
        if not os.path.exists(file_path):
            continue
        
        # Load the most recent sample from the node's data
        data = np.load(file_path)
        last_sample = torch.tensor(data['features'][-1], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(last_sample)
            risk_class = torch.argmax(output, dim=1).item()
            probs = torch.softmax(output, dim=1)[0]
            confidence = probs[risk_class].item() * 100

        # 4. Add Marker to Map
        html_popup = f"""
        <div style='font-family: Arial; width: 200px;'>
            <h4>{info['name']}</h4>
            <p><b>Status:</b> {labels[risk_class]}</p>
            <p><b>Confidence:</b> {confidence:.2f}%</p>
            <hr>
            <small>Source: P²MS-Andes Federated System</small>
        </div>
        """
        
        folium.Marker(
            location=info['coords'],
            popup=folium.Popup(html_popup, max_width=250),
            tooltip=f"Click for {info['name']} Risk Report",
            icon=folium.Icon(color=colors[risk_class], icon="info-sign")
        ).add_to(m)

    # 5. Save the Dashboard
    save_path = "data/processed/risk_map.html"
    m.save(save_path)
    print(f"✅ Dashboard generated successfully!")
    print(f"-> Open this file in your browser: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    generate_risk_map()