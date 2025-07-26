import bpy
import random
from mathutils import Vector

bl_info = {
    "name": "Simple Mushroom Simulation",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Mushroom",
    "description": "Simple mushroom substrate simulation",
    "category": "3D View",
}

# Simple simulation function
def run_simple_simulation():
    # Get parameters from the scene
    scene = bpy.context.scene
    lignin_content = scene.mushroom_lignin
    cellulose_content = scene.mushroom_cellulose
    moisture_content = scene.mushroom_moisture
    pasteurization_method = scene.mushroom_pasteurization
    mycelium_density = scene.mushroom_mycelium_density
    
    # Clear existing objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    # Simulation parameters
    lignin_content = 0.3
    cellulose_content = 0.5
    moisture_content = 0.7
    pasteurization_method = 'LIME'
    mycelium_density = 0.6
    
    # Print initial parameters to console
    print("\n" + "="*50)
    print("MUSHROOM SUBSTRATE SIMULATION - PARAMETERS")
    print("="*50)
    print(f"Substrate Composition:")
    print(f"  Lignin Content: {lignin_content:.2f}")
    print(f"  Cellulose Content: {cellulose_content:.2f}")
    print(f"  Moisture Content: {moisture_content:.2f}")
    print(f"\nPasteurization Method: {pasteurization_method}")
    print(f"Mycelium Density: {mycelium_density:.2f}")
    
    # Calculate conductivity (simplified formula)
    base_conductivity = 0.01
    moisture_factor = 10.0
    mycelium_factor = 5.0
    conductivity = base_conductivity + (moisture_content * moisture_factor * base_conductivity) + (mycelium_density * mycelium_factor * base_conductivity)
    
    # Calculate pH based on pasteurization method
    if pasteurization_method == 'LIME':
        ph = 12.0
    elif pasteurization_method == 'ASH':
        ph = 10.5
    elif pasteurization_method == 'SOAP':
        ph = 9.0
    elif pasteurization_method == 'BLEACH':
        ph = 11.0
    else:  # HEAT
        ph = 7.0
    
    # Print calculated results to console
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"pH Level: {ph:.2f}")
    print(f"Electrical Conductivity: {conductivity:.6f} Siemens/meter")
    
    # Calculate contaminant resistance (simplified)
    method_effectiveness = {
        'LIME': 0.9,
        'ASH': 0.8,
        'SOAP': 0.7,
        'BLEACH': 0.95,
        'HEAT': 0.85
    }
    contaminant_resistance = 0.5 * method_effectiveness[pasteurization_method]
    print(f"Contaminant Resistance: {contaminant_resistance:.2f}")
    
    # Create substrate (cube)
    bpy.ops.mesh.primitive_cube_add(size=1)
    substrate = bpy.context.active_object
    substrate.name = "Substrate"
    substrate.location = (0, 0, 0)
    substrate.scale = (5, 5, 2)
    
    # Create material for substrate - color based on parameters
    mat = bpy.data.materials.new(name="SubstrateMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Color based on lignin and moisture
    principled.inputs['Base Color'].default_value = (
        0.4 + lignin_content * 0.2,
        0.3 + cellulose_content * 0.2,
        0.1 + moisture_content * 0.2,
        1.0
    )
    principled.inputs['Roughness'].default_value = 1.0 - moisture_content * 0.5
    
    mat.node_tree.links.new(principled.outputs[0], output.inputs[0])
    
    if substrate.data.materials:
        substrate.data.materials[0] = mat
    else:
        substrate.data.materials.append(mat)
    
    # Create mycelium material (white)
    myc_mat = bpy.data.materials.new(name="MyceliumMaterial")
    myc_mat.use_nodes = True
    myc_nodes = myc_mat.node_tree.nodes
    for node in myc_nodes:
        myc_nodes.remove(node)
    
    myc_output = myc_nodes.new(type='ShaderNodeOutputMaterial')
    myc_principled = myc_nodes.new(type='ShaderNodeBsdfPrincipled')
    myc_principled.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)
    myc_principled.inputs['Roughness'].default_value = 0.3
    myc_mat.node_tree.links.new(myc_principled.outputs[0], myc_output.inputs[0])
    
    # Create simple mycelium (spheres) - number based on density
    num_spheres = int(20 * mycelium_density)
    for i in range(num_spheres):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        z = random.uniform(1, 3)
        
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(x, y, z))
        sphere = bpy.context.active_object
        sphere.name = f"Mycelium_{i}"
        sphere.data.materials.append(myc_mat)
    
    # Create electrodes
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, location=(-6, 0, 0))
    left_electrode = bpy.context.active_object
    left_electrode.name = "LeftElectrode"
    left_electrode.rotation_euler = (0, 1.5708, 0)  # 90 degrees in radians
    
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, location=(6, 0, 0))
    right_electrode = bpy.context.active_object
    right_electrode.name = "RightElectrode"
    right_electrode.rotation_euler = (0, 1.5708, 0)  # 90 degrees in radians
    
    # Create electrode material (metallic)
    elec_mat = bpy.data.materials.new(name="ElectrodeMaterial")
    elec_mat.use_nodes = True
    elec_nodes = elec_mat.node_tree.nodes
    for node in elec_nodes:
        elec_nodes.remove(node)
    
    elec_output = elec_nodes.new(type='ShaderNodeOutputMaterial')
    elec_principled = elec_nodes.new(type='ShaderNodeBsdfPrincipled')
    elec_principled.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
    elec_principled.inputs['Metallic'].default_value = 1.0
    elec_principled.inputs['Roughness'].default_value = 0.1
    elec_mat.node_tree.links.new(elec_principled.outputs[0], elec_output.inputs[0])
    
    left_electrode.data.materials.append(elec_mat)
    right_electrode.data.materials.append(elec_mat)
    
    # Create detailed text display
    bpy.ops.object.text_add(location=(0, 0, 5))
    text = bpy.context.active_object
    text.data.body = f"""Mushroom Substrate Simulation

Substrate Properties:
Lignin Content: {lignin_content:.2f}
Cellulose Content: {cellulose_content:.2f}
Moisture Content: {moisture_content:.2f}

Pasteurization Method: {pasteurization_method}
pH Level: {ph:.1f}

Mycelium Density: {mycelium_density:.2f}

Electrical Properties:
Conductivity: {conductivity:.6f} S/m"""
    
    text.data.size = 0.5
    text.rotation_euler = (1.5708, 0, 0)  # 90 degrees in X to make it readable from above
    
    # Set up camera for better view
    bpy.ops.object.camera_add(location=(0, -15, 10), rotation=(0.7, 0, 0))
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    
    # Add lighting
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    
    # At the end of the function, print a summary
    print("\n" + "="*50)
    print("SIMULATION COMPLETE")
    print("="*50)
    print(f"The substrate with {pasteurization_method} pasteurization has:")
    print(f"- pH level of {ph:.2f}")
    print(f"- Electrical conductivity of {conductivity:.6f} S/m")
    print(f"- Contaminant resistance of {contaminant_resistance:.2f}")
    print(f"- {num_spheres} mycelium growth points")
    print("="*50 + "\n")
    
    return f"Simulation completed successfully! Conductivity: {conductivity:.6f} S/m, pH: {ph:.1f}"

# Simple operator
class MUSHROOM_OT_simple_simulation(bpy.types.Operator):
    bl_idname = "mushroom.simple_simulation"
    bl_label = "Run Simple Mushroom Simulation"
    
    def execute(self, context):
        result = run_simple_simulation()
        self.report({'INFO'}, result)
        return {'FINISHED'}

# Simple panel
class MUSHROOM_PT_simple_panel(bpy.types.Panel):
    bl_label = "Mushroom Simulation"
    bl_idname = "MUSHROOM_PT_simple_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mushroom"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Substrate properties
        box = layout.box()
        box.label(text="Substrate Properties")
        box.prop(scene, "mushroom_lignin")
        box.prop(scene, "mushroom_cellulose")
        box.prop(scene, "mushroom_moisture")
        
        # Pasteurization method
        box = layout.box()
        box.label(text="Pasteurization")
        box.prop(scene, "mushroom_pasteurization")
        
        # Mycelium properties
        box = layout.box()
        box.label(text="Mycelium")
        box.prop(scene, "mushroom_mycelium_density")
        
        # Run simulation button
        layout.operator("mushroom.simple_simulation")

# Registration
classes = (
    MUSHROOM_OT_simple_simulation,
    MUSHROOM_PT_simple_panel,
)

def register():
    # Register classes
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register properties
    bpy.types.Scene.mushroom_lignin = bpy.props.FloatProperty(
        name="Lignin Content",
        description="Amount of lignin in substrate",
        min=0.0, max=1.0,
        default=0.3
    )
    bpy.types.Scene.mushroom_cellulose = bpy.props.FloatProperty(
        name="Cellulose Content",
        description="Amount of cellulose in substrate",
        min=0.0, max=1.0,
        default=0.5
    )
    bpy.types.Scene.mushroom_moisture = bpy.props.FloatProperty(
        name="Moisture Content",
        description="Amount of moisture in substrate",
        min=0.0, max=1.0,
        default=0.7
    )
    bpy.types.Scene.mushroom_mycelium_density = bpy.props.FloatProperty(
        name="Mycelium Density",
        description="Density of mycelium growth",
        min=0.0, max=1.0,
        default=0.6
    )
    bpy.types.Scene.mushroom_pasteurization = bpy.props.EnumProperty(
        name="Pasteurization Method",
        description="Method used to pasteurize the substrate",
        items=[
            ('LIME', "Lime Water", "Pasteurization using lime water"),
            ('ASH', "Ash Water", "Pasteurization using wood ash water"),
            ('SOAP', "Soap Solution", "Pasteurization using soap solution"),
            ('BLEACH', "Bleach Solution", "Pasteurization using diluted bleach"),
            ('HEAT', "Heat", "Traditional heat pasteurization")
        ],
        default='LIME'
    )

def unregister():
    # Unregister classes
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    # Unregister properties
    del bpy.types.Scene.mushroom_lignin
    del bpy.types.Scene.mushroom_cellulose
    del bpy.types.Scene.mushroom_moisture
    del bpy.types.Scene.mushroom_mycelium_density
    del bpy.types.Scene.mushroom_pasteurization

if __name__ == "__main__":
    register() 