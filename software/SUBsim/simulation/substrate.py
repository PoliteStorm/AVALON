class SubstrateParameters:
    def __init__(self):
        # Substrate composition
        self.substrate_type = "HARDWOOD"  # Type of base substrate
        self.lignin_content = 0.3  # 0-1 scale
        self.cellulose_content = 0.5  # 0-1 scale
        self.moisture_content = 0.7  # 0-1 scale
        self.nitrogen_content = 0.02  # 0-1 scale
        self.initial_ph = 6.0  # pH scale 0-14
        
        # Pasteurization method
        self.pasteurization_method = 'HEAT'  # 'HEAT', 'LIME', 'HYDROGEN_PEROXIDE', 'STEAM', 'BLEACH', 'NONE'
        self.pasteurization_duration = 24.0  # hours
        self.pasteurization_concentration = 0.5  # concentration of pasteurization agent
        self.pasteurization_temperature = 65.0  # °C
        
        # Calcium supplementation
        self.calcium_supplement = 'NONE'  # 'NONE', 'OYSTER_SHELL', 'LIMESTONE', 'GYPSUM'
        self.calcium_content = 0.0  # 0-1 scale
        self.calcium_calcination = False  # Whether calcium supplement is calcinated
        
        # Mycelium properties
        self.mushroom_species = "OYSTER"  # Type of mushroom
        self.mycelium_density = 0.5  # 0-1 scale
        self.growth_rate = 0.05  # growth per time unit
        self.spawn_rate = 0.1  # 0-1 scale
        self.mycelium_age = 14.0  # days
        self.mycelium_species = "OYSTER"  # Species of mushroom
        
        # Environmental conditions
        self.incubation_temperature = 24.0  # °C
        self.humidity = 85.0  # %
        
        # Electrical properties
        self.base_conductivity = 0.01  # Siemens/meter
        self.moisture_conductivity_factor = 10.0  # How much moisture increases conductivity
        self.mycelium_conductivity_factor = 5.0  # How much mycelium increases conductivity

def calculate_conductivity(params):
    """Calculate electrical conductivity based on substrate parameters"""
    # Base conductivity from substrate
    conductivity = params.base_conductivity
    
    # Add effect of moisture (exponential relationship)
    moisture_effect = params.moisture_content ** 2 * params.moisture_conductivity_factor * params.base_conductivity
    
    # Add effect of mycelium (linear relationship)
    mycelium_effect = params.mycelium_density * params.mycelium_conductivity_factor * params.base_conductivity
    
    # Add effect of pH (ions increase conductivity)
    ph_factor = 1.0
    if hasattr(params, 'initial_ph'):
        # pH away from neutral (7) increases conductivity due to more free ions
        ph_factor = 1.0 + abs(params.initial_ph - 7.0) * 0.2
    
    # Add effect of pasteurization method (some methods add ions)
    method_factor = 1.0
    if params.pasteurization_method == 'LIME':
        method_factor = 2.0  # Lime adds calcium ions
    elif params.pasteurization_method == 'BLEACH':
        method_factor = 1.5  # Bleach adds sodium and chloride ions
    
    # Calculate final conductivity
    conductivity = (conductivity + moisture_effect + mycelium_effect) * ph_factor * method_factor
    
    return conductivity

def calculate_ph(params):
    """Calculate the pH of the substrate based on parameters"""
    # Base pH from substrate type
    base_ph = {
        "HARDWOOD": 5.5,
        "SOFTWOOD": 5.0,
        "STRAW": 7.0,
        "COFFEE_GROUNDS": 6.0,
        "COMPOST": 7.5,
        "COCO_COIR": 6.0,
        "OYSTER_SHELL_MIX": 7.8  # Oyster shell increases pH significantly
    }.get(params.substrate_type, 6.0)
    
    # Adjust for initial pH
    base_ph = (base_ph + params.initial_ph) / 2
    
    # Calcium supplements affect pH
    if params.calcium_supplement == 'OYSTER_SHELL':
        # Oyster shell is a strong pH buffer, raising pH toward 7.5-8.0
        ph_increase = 0.5 + (1.5 * params.calcium_content)
        if params.calcium_calcination:
            ph_increase *= 1.5  # Calcinated oyster shell has stronger pH effect
        base_ph += ph_increase
    elif params.calcium_supplement == 'LIMESTONE':
        base_ph += 0.3 + (1.0 * params.calcium_content)
    elif params.calcium_supplement == 'GYPSUM':
        base_ph += 0.1 + (0.3 * params.calcium_content)
    
    # Pasteurization method affects pH
    pasteurization_effect = {
        'HEAT': 0,
        'LIME': 1.5,
        'HYDROGEN_PEROXIDE': -0.5,
        'STEAM': 0.2,
        'BLEACH': 0.8,
        'NONE': 0
    }.get(params.pasteurization_method, 0)
    
    # Adjust pH based on pasteurization method and duration
    base_ph += pasteurization_effect * (params.pasteurization_duration / 24) * params.pasteurization_concentration
    
    # Ensure pH is within realistic bounds
    return max(3.0, min(10.0, base_ph))

def calculate_contaminant_resistance(params):
    """Calculate resistance to contaminants based on parameters"""
    method_effectiveness = {
        'HEAT': 0.85,
        'LIME': 0.9,
        'HYDROGEN_PEROXIDE': 0.8,
        'STEAM': 0.95,
        'BLEACH': 0.95,
        'NONE': 0.1
    }
    
    # Base resistance from pasteurization method
    base_resistance = method_effectiveness.get(params.pasteurization_method, 0.5)
    
    # Duration factor (longer pasteurization = better resistance, up to a point)
    duration_factor = min(1.0, params.pasteurization_duration / 24.0)
    
    # Concentration factor
    concentration_factor = params.pasteurization_concentration
    
    # pH factor (extreme pH inhibits contaminants)
    ph = calculate_ph(params)
    ph_factor = 1.0
    if ph > 11.0 or ph < 4.0:
        ph_factor = 1.2  # Very high or low pH inhibits many contaminants
    elif ph > 9.0 or ph < 5.0:
        ph_factor = 1.1  # Moderately extreme pH helps somewhat
    
    # Moisture factor (too wet = more contamination risk)
    moisture_factor = 1.0
    if params.moisture_content > 0.8:
        moisture_factor = 0.8  # Too wet
    elif params.moisture_content < 0.5:
        moisture_factor = 0.9  # Too dry
    
    # Calculate final resistance (capped at 1.0)
    resistance = base_resistance * duration_factor * concentration_factor * ph_factor * moisture_factor
    
    return min(1.0, resistance)

def calculate_growth_potential(params):
    """Calculate the growth potential of mycelium based on substrate parameters"""
    # Base growth potential from substrate type
    base_potential = {
        "HARDWOOD": 0.8,
        "SOFTWOOD": 0.6,
        "STRAW": 0.7,
        "COFFEE_GROUNDS": 0.9,
        "COMPOST": 0.75,
        "COCO_COIR": 0.65,
        "OYSTER_SHELL_MIX": 0.85  # Oyster shell provides excellent calcium and pH buffering
    }.get(params.substrate_type, 0.7)
    
    # Adjust for moisture content (optimal around 60-70%)
    moisture_factor = 1.0 - 2.0 * abs(params.moisture_content - 0.65)
    
    # Adjust for lignin content (lower is better for most species)
    lignin_factor = 1.0 - params.lignin_content
    
    # Adjust for cellulose content (higher is better)
    cellulose_factor = 0.5 + 0.5 * params.cellulose_content
    
    # Adjust for nitrogen content (optimal around 1-2%)
    nitrogen_factor = 1.0 - 10.0 * abs(params.nitrogen_content - 0.015)
    
    # Adjust for pH (optimal depends on species)
    ph = calculate_ph(params)
    optimal_ph = {
        "OYSTER": 6.0,
        "SHIITAKE": 5.5,
        "LIONS_MANE": 6.5,
        "REISHI": 5.5,
        "BUTTON": 6.5,
        "ENOKI": 6.0,
        "MAITAKE": 6.0,
        "CORDYCEPS": 6.5
    }.get(params.mycelium_species, 6.0)
    ph_factor = 1.0 - 0.3 * abs(ph - optimal_ph)
    
    # Calcium supplementation effect
    calcium_factor = 1.0
    if params.calcium_supplement == 'OYSTER_SHELL':
        # Research shows oyster shell significantly improves mycelium growth
        calcium_boost = 0.2 * params.calcium_content
        if params.calcium_calcination:
            calcium_boost *= 1.5  # Calcinated oyster shell is more bioavailable
        calcium_factor += calcium_boost
    elif params.calcium_supplement in ['LIMESTONE', 'GYPSUM']:
        calcium_factor += 0.1 * params.calcium_content
    
    # Calculate overall growth potential
    growth_potential = base_potential * moisture_factor * lignin_factor * cellulose_factor * nitrogen_factor * ph_factor * calcium_factor
    
    # Ensure growth potential is within realistic bounds
    return max(0.0, min(1.0, growth_potential))

def calculate_electrical_properties(params):
    """Calculate various electrical properties for unconventional computing"""
    # Basic conductivity
    conductivity = calculate_conductivity(params)
    
    # Resistivity (inverse of conductivity)
    resistivity = 1.0 / conductivity if conductivity > 0 else float('inf')
    
    # Impedance (complex resistance, simplified model)
    # Increases with mycelium density due to capacitive effects
    impedance = resistivity * (1.0 + 0.2 * params.mycelium_density)
    
    # Dielectric constant (relative permittivity)
    # Mostly affected by water content
    dielectric = 4.0 + 16.0 * params.moisture_content
    
    # Signal propagation speed (relative to vacuum)
    # Decreases with higher dielectric constant
    propagation_speed = 1.0 / (dielectric ** 0.5)
    
    return {
        'conductivity': conductivity,  # S/m
        'resistivity': resistivity,    # Ω·m
        'impedance': impedance,        # Ω·m (simplified)
        'dielectric': dielectric,      # unitless
        'propagation_speed': propagation_speed  # relative to c
    } 