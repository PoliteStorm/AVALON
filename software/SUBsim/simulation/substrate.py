class SubstrateParameters:
    def __init__(self):
        # Substrate composition
        self.lignin_content = 0.3  # 0-1 scale
        self.cellulose_content = 0.5  # 0-1 scale
        self.moisture_content = 0.7  # 0-1 scale
        
        # Pasteurization method
        self.pasteurization_method = 'LIME'  # 'LIME', 'ASH', 'SOAP', 'BLEACH', 'HEAT'
        self.pasteurization_duration = 24.0  # hours
        self.pasteurization_concentration = 0.5  # concentration of pasteurization agent
        
        # Mycelium properties
        self.mycelium_density = 0.6  # 0-1 scale
        self.growth_rate = 0.05  # growth per time unit
        
        # Electrical properties
        self.base_conductivity = 0.01  # Siemens/meter
        self.moisture_conductivity_factor = 10.0  # How much moisture increases conductivity
        self.mycelium_conductivity_factor = 5.0  # How much mycelium increases conductivity

def calculate_conductivity(params):
    """Calculate electrical conductivity based on substrate parameters"""
    # Base conductivity from substrate
    conductivity = params.base_conductivity
    
    # Add effect of moisture
    conductivity += params.moisture_content * params.moisture_conductivity_factor * params.base_conductivity
    
    # Add effect of mycelium
    conductivity += params.mycelium_density * params.mycelium_conductivity_factor * params.base_conductivity
    
    return conductivity

def calculate_ph(params):
    """Calculate pH based on pasteurization method"""
    if params.pasteurization_method == 'LIME':
        # Lime water creates highly alkaline environment
        return 12.0 + params.pasteurization_concentration * 0.5
    elif params.pasteurization_method == 'ASH':
        # Wood ash is also alkaline but typically less than lime
        return 10.5 + params.pasteurization_concentration * 1.0
    elif params.pasteurization_method == 'SOAP':
        # Soap is mildly alkaline
        return 9.0 + params.pasteurization_concentration * 0.5
    elif params.pasteurization_method == 'BLEACH':
        # Bleach is highly alkaline
        return 11.0 + params.pasteurization_concentration * 1.0
    else:  # HEAT
        # Heat pasteurization doesn't significantly change pH
        return 7.0

def calculate_contaminant_resistance(params):
    """Calculate resistance to contaminants based on parameters"""
    method_effectiveness = {
        'LIME': 0.9,
        'ASH': 0.8,
        'SOAP': 0.7,
        'BLEACH': 0.95,
        'HEAT': 0.85
    }
    
    base_resistance = 0.5
    duration_factor = min(1.0, params.pasteurization_duration / 24.0)
    concentration_factor = params.pasteurization_concentration
    
    return base_resistance * method_effectiveness[params.pasteurization_method] * duration_factor * concentration_factor 