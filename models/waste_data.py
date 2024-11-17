import pandas as pd
import json

class WasteData:
    def __init__(self):
        # Load dumpsters data
        self.dumpsters_df = pd.read_csv('data/wastedumps.csv')
        
        # Load companies data
        self.companies_df = pd.read_csv('data/companies.csv')
        
        # Convert string representations of lists to actual lists
        self.companies_df['raw_materials_needed'] = self.companies_df['raw_materials_needed'].apply(eval)
        self.companies_df['waste_produced'] = self.companies_df['waste_produced'].apply(eval)
    
    def get_all_materials(self):
        """Get list of all possible materials"""
        materials = []
        for col in self.dumpsters_df.columns:
            if col.startswith('accepts_'):
                materials.append(col)
        return materials