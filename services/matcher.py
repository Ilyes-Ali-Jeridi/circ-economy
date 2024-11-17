from utils.distance import haversine
import networkx as nx
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class WasteMatcher:
    def __init__(self, waste_data):
        self.waste_data = waste_data
        self.MAX_DISTANCE = 50  # Maximum distance in kilometers
        self.scaler = MinMaxScaler()
        self.rnn_model = self.build_rnn_model()
        self.initialize_model()
        
    def build_rnn_model(self):
        """Builds an RNN model for waste flow prediction."""
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(None, 4), return_sequences=True),
            LSTM(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Output: Predicted waste flow
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def prepare_data(self, company_data):
        """Prepare data for RNN prediction."""
        # Extract relevant features
        features = []
        for _, company in self.waste_data.companies_df.iterrows():
            distance = haversine(
                company_data["latitude"],
                company_data["longitude"],
                company["latitude"],
                company["longitude"]
            )
            
            # Features: [distance, num_materials, num_wastes, quantity]
            feature_vector = [
                distance,
                len(company["raw_materials_needed"]),
                len(company["waste_produced"]),
                company_data.get("quantity", 0)
            ]
            features.append(feature_vector)
            
        features = np.array(features)
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Reshape for RNN: (samples, timesteps, features)
        sequence_length = 5
        X = []
        for i in range(len(features_scaled) - sequence_length + 1):
            X.append(features_scaled[i:i + sequence_length])
        
        return np.array(X)
    
    def initialize_model(self):
        """Initialize the model with some synthetic data."""
        # Generate synthetic training data
        num_samples = 100
        sequence_length = 5
        num_features = 4
        
        X_train = np.random.rand(num_samples, sequence_length, num_features)
        y_train = np.random.rand(num_samples, 1)  # Random target values
        
        # Train the model on synthetic data
        self.rnn_model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=16,
            verbose=0
        )
    
    def predict_waste_flow(self, company_data):
        """Predicts future waste flow for the given company."""
        X = self.prepare_data(company_data)
        if len(X) > 0:
            prediction = self.rnn_model.predict(X, verbose=0)
            return float(np.mean(prediction))  # Average predictions
        return 0.0
        
    def find_matches(self, company_data):
        """Find matching dumpsters and companies for given company data."""
        matches = {
            "dumpsters": [],
            "companies": [],
            "routes": {},
            "predicted_waste_flow": None
        }

        # Predict waste flow for the company
        predicted_waste_flow = self.predict_waste_flow(company_data)
        matches["predicted_waste_flow"] = predicted_waste_flow

        # Create a graph for routing
        G = nx.Graph()
        company_pos = (company_data["latitude"], company_data["longitude"])
        G.add_node("source", pos=company_pos)

        # Match with dumpsters within MAX_DISTANCE
        for _, dumpster in self.waste_data.dumpsters_df.iterrows():
            distance = haversine(
                company_data["latitude"],
                company_data["longitude"],
                dumpster["latitude"],
                dumpster["longitude"]
            )

            if distance <= self.MAX_DISTANCE:
                dumpster_materials = [col for col in self.waste_data.dumpsters_df.columns
                                    if col.startswith("accepts_") and dumpster[col] == True]

                if any(material in company_data["raw_materials_needed"] for material in dumpster_materials):
                    matches["dumpsters"].append({
                        "id": dumpster["id"],
                        "name": dumpster["name"],
                        "distance_km": distance,
                        "materials": dumpster_materials,
                        "location": f"{dumpster['address_number']} {dumpster['street']}, {dumpster['city']}",
                        "hours": dumpster["hours_of_operation"],
                        "notes": dumpster["notes"],
                        "latitude": dumpster["latitude"],
                        "longitude": dumpster["longitude"]
                    })

                    # Add to routing graph with predicted flow as weight factor
                    node_id = f"dumpster_{dumpster['id']}"
                    G.add_node(node_id, pos=(dumpster["latitude"], dumpster["longitude"]))
                    # Adjust edge weight based on predicted flow
                    edge_weight = distance * (1 + predicted_waste_flow)
                    G.add_edge("source", node_id, weight=edge_weight)

        # Match with companies within MAX_DISTANCE
        for _, other_company in self.waste_data.companies_df.iterrows():
            distance = haversine(
                company_data["latitude"],
                company_data["longitude"],
                other_company["latitude"],
                other_company["longitude"]
            )

            if distance <= self.MAX_DISTANCE:
                matching_materials = [waste for waste in other_company["waste_produced"]
                                   if waste in company_data["raw_materials_needed"]]

                if matching_materials:
                    matches["companies"].append({
                        "id": other_company["company_id"],
                        "name": other_company["company_name"],
                        "distance_km": distance,
                        "materials": matching_materials,
                        "latitude": other_company["latitude"],
                        "longitude": other_company["longitude"]
                    })

                    # Add to routing graph with predicted flow as weight factor
                    node_id = f"company_{other_company['company_id']}"
                    G.add_node(node_id, pos=(other_company["latitude"], other_company["longitude"]))
                    edge_weight = distance * (1 + predicted_waste_flow)
                    G.add_edge("source", node_id, weight=edge_weight)

        # Calculate shortest paths considering predicted flow
        if len(G.nodes) > 1:
            for node in G.nodes:
                if node != "source":
                    try:
                        path = nx.shortest_path(G, "source", node, weight="weight")
                        path_coords = [G.nodes[n]["pos"] for n in path]
                        matches["routes"][node] = path_coords
                    except nx.NetworkXNoPath:
                        continue

        # Sort matches by optimized score (distance and predicted flow)
        matches["dumpsters"].sort(key=lambda x: x["distance_km"] * (1 + predicted_waste_flow))
        matches["companies"].sort(key=lambda x: x["distance_km"] * (1 + predicted_waste_flow))

        return matches