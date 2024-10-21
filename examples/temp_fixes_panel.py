"""
Rayleigh-Benard Convection Flow
- copied from rayleigh-benard-chorin-dolfinx-dimensional.py by Ethan Young on 8/29 and fixed a few bugs:
    - switched negative to positive sign on viscosity term in momentum equation
    - added pressure term into momentum equation
    - pinned pressure by adding pressure BC
    - switched solvers back to GMRES and JACOBI
- copied from Walid Arsalene's FEniCS code, adapted to FEniCSx by Brooke Stanislawski
- uses the governing equations from Chorin 1968 (Oberbeck-Boussinesq approximation of the Navier-Stokes equations)
- includes the option to model the classical convective flow in an empty domain or to include a heated pv panel at the center of the domain
"""

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

import gmsh
from dolfinx import io
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
)

# from dolfinx.io import VTXWriter
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, locate_entities

# from dolfinx.plot import vtk_mesh
from ufl import (
    FacetNormal,
    FiniteElement,
    Identity,
    TestFunction,
    TrialFunction,
    VectorElement,
    div,
    dot,
    ds,
    dx,
    inner,
    lhs,
    nabla_grad,
    grad,
    rhs,
    sym,
    sqrt,
    CellDiameter,
)

# ================================================================
# Inputs
# ================================================================

x_min = 0.0
x_max = 1.2 # 0.4 #1.0

y_min = 0.0
y_max = 0.4 #1.0 #3.0

# h = 0.05
nx = 120 # 50 # 100 # 100 # 50  # 150 # int((x_max - x_min)/h)
ny = 40 # 50 # 100 # 150 #.5*.02/.1 30 # 10 # 50  # 50 # int((y_max - y_min)/h)

# values from Example 9.1 in Incropera
# T0_top_wall = 25+273.15 # 300.0 # 300.000 # 0
# T0_bottom_wall = 70.+273.15 # 300.00109 # 0 #1

# T_film = 300.0 # K
# T_delta = 0.0 # delta from T_film
# T0_top_wall = T_film - T_delta
# T0_bottom_wall = T_film + T_delta

# # heated channel
# T_ambient = 300.0
# T0_top_wall = 350.0
# T0_bottom_wall = 350.0

# flow over a flat plate
T_ambient = 300.0
T0_top_wall = T_ambient
T0_bottom_wall = T_ambient + 20.0
T0_pv_panel = T_ambient + 20.0 # only used if pv_panel_flag == True


# # uniform inflow
# inflow = 'uniform'
# u0 = 1.0

# loglaw influw
inflow = 'loglaw'
u_hub = 1.0
z_hub = 0.12
z0 = 0.005
d0 = 0.0 # 0.65*z_hub


# Gasteuil et al 2007
# def_hot_wall = 100.0 # 10.1 # 100.0
# T0_top_wall = -def_hot_wall #19+273.15 # 320.50 # 300.000 # 0
# T0_bottom_wall = def_hot_wall # 39.3+273.15 # 320.5007685957619775737  #300.00109 # 0 #1
# def_hot_wall = 100.0
# T0_top_wall = -def_hot_wall #19+273.15 # 320.50 # 300.000 # 0
# T0_bottom_wall = def_hot_wall # 39.3+273.15 # 320.5007685957619775737  #300.00109 # 0 #1

# T0_top_wall = 39+273.15 # 320.50 # 300.000 # 0
# T0_bottom_wall = 19+273.15 # 320.5007685957619775737  #300.00109 # 0 #1


# deltaT = T0_bottom_wall - T0_top_wall
# T_avg = (T0_top_wall+T0_bottom_wall)/2.
# T_f = T_avg # (deltaT/2.)+T0_top_wall
# ^ these are equivalent

T_f = T_ambient

stabilizing = False
pv_panel_flag = True  # empty domain or with a pv panel in the center?

save_fn = 'temp_panel'
t_final = 2.0 # 10.0 # 20.0 # 120.0 #1.0 # 10.0 #0.4 # 0.003 # 0.1  # 0.5 # 0.5 #0.1 # 0.000075
dt_num = 0.01 # 0.01 #0.001
# ================================================================
# Build Mesh
# ================================================================

if pv_panel_flag:
    # pass

    comm = MPI.COMM_WORLD

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gmsh_model = gmsh.model()
    gmsh_model.add("domain")
    gmsh_model.setCurrent("domain")

    ndim = 2

    domain_width = x_max - x_min #3.0  # box width
    domain_height = y_max - y_min # 1.0  #  # box height

    domain_id = gmsh_model.occ.addRectangle(
        0, 0, 0, domain_width, domain_height
    )  # Notice this spans from [0, x_max], [0, y_max], your BCs may need adjustment
    domain_tag = (ndim, domain_id)

    panel_width = 0.1 # 0.5  # Chord length, or width
    panel_height = 0.03 # 0.05  # Sets the panel thickness, really
    panel_angle = np.radians(
        30
    )  # Sets the panel rotation (argument must be radians for gmsh)

    panel_id = gmsh_model.occ.addRectangle(
        -0.5 * panel_width, -0.5 * panel_height, 0, panel_width, panel_height
    )
    panel_tag = (ndim, panel_id)

    # Rotate the panel and shift it into its correct position
    gmsh_model.occ.rotate([panel_tag], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, panel_angle)
    gmsh_model.occ.translate([panel_tag], 0.5 * domain_width, 0.3 * domain_height, 0.0)

    # Cookie cutter step, domain = domain - panel, is how to read this
    gmsh_model.occ.cut([domain_tag], [panel_tag])

    gmsh_model.occ.synchronize()

    all_pts = gmsh_model.occ.getEntities(0)

    l_characteristic = 1.0/nx # 0.05  # Sets the characteristic size of the cells
    gmsh_model.mesh.setSize(all_pts, l_characteristic)

    vol_tag_list = gmsh_model.occ.getEntities(ndim)

    for vol_tag in vol_tag_list:
        vol_id = vol_tag[1]
        gmsh_model.add_physical_group(ndim, [vol_id], vol_id)

    # Generate the mesh
    gmsh_model.mesh.generate(ndim)

    mesh, mt, ft = io.gmshio.model_to_mesh(gmsh_model, comm, 0, gdim=2)

else:
    # create an empty domain mesh
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([x_min, y_min]), np.array([x_max, y_max])],
        [nx, ny],
        CellType.triangle,
    )


# Two key physical parameters are the Rayleigh number (Ra), which
# measures the ratio of energy from buoyant forces to viscous
# dissipation and heat conduction and the
# Prandtl number (Pr), which measures the ratio of viscosity to heat
# conduction.

# ================================================================
# Define Constants
# ================================================================

# calc alpha from Incropera for water at 300 K
# g_f = -9.81
# beta_f = 2.76e-4
# nu_f = 1.003*10**(-3)
# k_f = 613*10**(-3) # W/m*K
# rho_f = 993.88 #998.57 # kg/m3
# cp_f = 4.179*1000 # J/kg*K

# calc alpha from Incropera for air at 300 K
# g_f = -9.81
# beta_f = 1/300.0 # [1/K]
# nu_f = 15.89e-6 # 0.01 # 15.89e-6 # 0.01 # [m2/s]
# # k_f = 0.0263 # W/m*K
# alpha_f = 22.5/10**6 # m2/s
# rho_f = 1.1314 # kg/m3
# # cp_f = 1.004*1000 # J/kg*K

# # alpha_f = k_f/(rho_f*cp_f) # m2/s

# from Gasteuil et al 2007
# g_f = -98.1
# # beta_f = 2.95e-4
# beta_f = 0.3
# nu_f = 8.17e-7
# alpha_f = 1.48e-7
# rho_f = 993.88 #998.57 # kg/m3

# from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html
g_f = -98.1
beta_f = 0.01
alpha_f = 0.01 #22.5/10**6 # m2/s
rho_f = 1.0 # kg/m3
mu_f = 0.01

# mu_f = nu_f * rho_f

# Ra = (g_f*beta_f/(nu_f*alpha_f))*(T0_bottom_wall-T0_top_wall)*(y_max-y_min)

print('alpha = ', alpha_f)
print('mu = ', mu_f)
# print('Ra = {:.2E}'.format(Ra))

# Ra = Constant(1e8)
# Ra = Constant(mesh, PETSc.ScalarType(1e5))
# # Ra = Constant(mesh, PETSc.ScalarType(2500))

# Pr = Constant(mesh, PETSc.ScalarType(0.7))

# g = Constant(mesh, PETSc.ScalarType((0, 1)))
g = Constant(mesh, PETSc.ScalarType((0, g_f))) # negative? YES
# g = Constant(mesh, PETSc.ScalarType((0, -9.81))) # negative? YES
# g = Constant(mesh, PETSc.ScalarType((0, -98.1)))
# g = Constant(mesh, PETSc.ScalarType((0, -9800000.1)))
# g = Constant(mesh, PETSc.ScalarType((0, 0)))

# nu = Constant(mesh, PETSc.ScalarType(1))
# nu = Constant(mesh, PETSc.ScalarType(15.89e-6)) # kinematic viscosity

# beta = Constant(mesh, PETSc.ScalarType(1/300)) # thermal expansion coefficient # Incroprera
# beta = Constant(mesh, PETSc.ScalarType(1./T_f)) # thermal expansion coefficient # Incroprera
# beta = Constant(mesh, PETSc.ScalarType(1/T0_bottom_wall)) # thermal expansion coefficient # Incroprera
# beta = Constant(mesh, PETSc.ScalarType(1))

# alpha = Constant(mesh, PETSc.ScalarType(22.5e-6)) # thermal diffusivity [m2/s]
# alpha = Constant(mesh, PETSc.ScalarType(26.2e-6)) # thermal diffusivity [m2/s]
# k = Constant(mesh, PETSc.ScalarType(26.3e-3)) # thermal conductivity [W/mK]

# rho = Constant(mesh, PETSc.ScalarType(1.1614)) # density [kg/m3]
# rho = Constant(mesh, PETSc.ScalarType(1.0782)) # density [kg/m3]

# mu = Constant(mesh, PETSc.ScalarType(184.6e-7)) # dynamic viscosity [Ns/m2]
# mu = Constant(mesh, PETSc.ScalarType(196.4e-7)) # dynamic viscosity [Ns/m2]

# from Gasteuil et al 2007 and Incropera
# beta = Constant(mesh, PETSc.ScalarType(2.95e-4)) # [1/K] thermal expansion coefficient (also alpha)
# beta = Constant(mesh, PETSc.ScalarType(2.76e-4)) # [1/K] thermal expansion coefficient
beta = Constant(mesh, PETSc.ScalarType(beta_f)) # [1/K] thermal expansion coefficient
# beta = Constant(mesh, PETSc.ScalarType(0.01)) # [1/K] thermal expansion coefficient
# alpha = Constant(mesh, PETSc.ScalarType(1.48e-4)) # thermal diffusivity [m2/s]
# alpha = Constant(mesh, PETSc.ScalarType(1.48e-7)) # thermal diffusivity [m2/s]
alpha = Constant(mesh, PETSc.ScalarType(alpha_f)) # thermal diffusivity [m2/s]
# alpha = Constant(mesh, PETSc.ScalarType(0.01)) # thermal diffusivity [m2/s] # this is 1e-4
# rho = Constant(mesh, PETSc.ScalarType(993.88)) # density [kg/m3]
rho = Constant(mesh, PETSc.ScalarType(rho_f)) # density [kg/m3]
# rho = Constant(mesh, PETSc.ScalarType(1.0)) # density [kg/m3]
# rho = Constant(mesh, PETSc.ScalarType(10.0)) # density [kg/m3]
# cp = Constant(mesh, PETSc.ScalarType(cp_f)) # [J/kg*K]
# k = Constant(mesh, PETSc.ScalarType(k_f)) # [W/m*K]
# mu = Constant(mesh, PETSc.ScalarType(812.e-6)) # dynamic viscosity [Ns/m2]
# mu = Constant(mesh, PETSc.ScalarType(8.55e-4)) # dynamic viscosity [Ns/m2]
# mu = Constant(mesh, PETSc.ScalarType(0.01)) # dynamic viscosity [Ns/m2] # Re = 100
mu = Constant(mesh, PETSc.ScalarType(mu_f)) # dynamic viscosity [Ns/m2] # Re = 100
# mu = Constant(mesh, PETSc.ScalarType(0.0025)) # dynamic viscosity [Ns/m2] # Re = 400
# nu = Constant(mesh, PETSc.ScalarType(8.6026e-07)) # kinematic viscosity [m3/kg]


dt = Constant(mesh, PETSc.ScalarType(dt_num))

# ================================================================
# Build Function Spaces and Functions
# ================================================================

v_cg2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
q_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

V = FunctionSpace(mesh, v_cg2)  # velocity
Q = FunctionSpace(mesh, q_cg1)  # pressure
S = FunctionSpace(mesh, q_cg1)  # temperature

# velocity
u = TrialFunction(V)
v = TestFunction(V)
u_n = Function(V)
u_n.name = "u_n"
u_ = Function(V)

# pressure
p = TrialFunction(Q)
q = TestFunction(Q)
p_n = Function(Q)
p_n.name = "p_n"
p_ = Function(Q)

# temperature
theta = TrialFunction(S)
s = TestFunction(S)
T_n = Function(S)  # for outputting T, calculated from theta for each timestep
T_n.name = "T_n"
T_ = Function(S)
# T_r = Function(S)  # for outputting T, calculated from theta for each timestep
# theta_n = Function(S)
# theta_n.name = "theta_n"
# theta_ = Function(S)

# %% ================================================================
# Build Boundary Conditions
# ================================================================
class InletVelocity():
    # copied from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
    def __init__(self):
        pass

    def __call__(self, x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        print('x.shape vel = ', x.shape)
        if inflow == 'uniform':
            values[0] = u0

        elif inflow == 'loglaw':
            # values[0] = ((u_hub) * np.log(((x[1]) - d0) / z0) / (np.log((z_hub - d0) / z0)))
            values[0] = ((u_hub) * np.log(((x[1]) - d0) / z0) / (np.log((z_hub - d0) / z0)))

            print(values[0]) # might need fixing close to the ground

            # h_panel = 0.15
            # u_star = 0.45
            # kappa = 0.41
            # z0 = 0.005
            # d0 = 0.65*h_panel
            # values[0] = (u_star/kappa)*(np.log((x[1]-d0)/z0))

            # exit()
        return values
    
class LowerWallTemperature():
    # copied from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
    def __init__(self):
        pass

    def __call__(self, x):
        values = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
        print('x.shape temp = ', x.shape)
        # print('x.shape[0] = ', x.shape[0])
        # print('x.shape[1] = ', x.shape[1])
        # exit()
        # values[0] = T0_bottom_wall
        x0 = 0.75 * x_max # start of ramp down
        values[0] = (T0_bottom_wall + ((x[0]-x0) / x_max) * (T_ambient - T0_bottom_wall))
        print(values[0]) # might need fixing close to the ground
        # exit()

        return values

def left_wall(x):
    return np.isclose(x[0], x_min)

def right_wall(x):
    return np.isclose(x[0], x_max)

def bottom_wall(x):
    return np.isclose(x[1], y_min)

def top_wall(x):
    return np.isclose(x[1], y_max)

def bottom_left_corner(x):
    return np.logical_and(np.isclose(x[1], y_min), np.isclose(x[0], x_min))

def upper_right_corner(x):
    return np.logical_and(np.isclose(x[1], y_max), np.isclose(x[0], x_max))

def internal_boundaries(x):
    tol = 1e-3
    x_test = np.logical_and(x_min + tol < x[0], x[0] < x_max - tol)
    y_test = np.logical_and(y_min + tol < x[1], x[1] < y_max - tol)
    return np.logical_and(x_test, y_test)


# pin pressure with bc at corner
# bottom_left_corner_pressure_dofs = locate_dofs_geometrical(Q, bottom_left_corner)
# bcp_bottom_left_corner = dirichletbc(0.0, bottom_left_corner_pressure_dofs, Q)
# bcp = [bcp_bottom_left_corner]
# # bcp = []

# Velocity Boundary Conditions
# Inlet
upper_cells = locate_entities(mesh, mesh.geometry.dim, lambda x: x[1] > d0 + z0)
u_inlet = Function(V)
u_inlet.interpolate(lambda x: np.zeros((mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType))
left_wall_dofs = locate_dofs_geometrical(V, left_wall)
inlet_velocity = InletVelocity()
u_inlet.interpolate(inlet_velocity, upper_cells)
bcu_inflow = dirichletbc(u_inlet, left_wall_dofs)

# print(np.shape(u_inlet.x.array[:]))
# exit()

# left_wall_dofs = locate_dofs_geometrical(V, left_wall)
u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
# u_inflow = np.array((1,0), dtype=PETSc.ScalarType) # ux, uy = 1, 0

# u_lid = np.array((1,0), dtype=PETSc.ScalarType) # ux, uy = 1, 0
# u_noslip = np.array((10,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
# bcu_left_wall = dirichletbc(u_noslip, left_wall_dofs, V)
# bcu_left_wall = dirichletbc(PETSc.ScalarType(2.0), left_wall_dofs, V)

# right_wall_dofs = locate_dofs_geometrical(V, right_wall)
# bcu_right_wall = dirichletbc(u_noslip, right_wall_dofs, V)
# bcu_right_wall = dirichletbc(u_inflow, right_wall_dofs, V)

bottom_wall_dofs = locate_dofs_geometrical(V, bottom_wall)
bcu_bottom_wall = dirichletbc(u_noslip, bottom_wall_dofs, V)

# top_wall_dofs = locate_dofs_geometrical(V, top_wall)
# bcu_top_wall = dirichletbc(u_noslip, top_wall_dofs, V)

# slip at top wall
top_wall_entities = locate_entities_boundary(mesh, mesh.geometry.dim-1, top_wall)
top_wall_dofs = locate_dofs_topological(V.sub(1), mesh.geometry.dim-1, top_wall_entities)
zero_scalar = Constant(mesh, PETSc.ScalarType(0.0))
bcu_top_wall = dirichletbc(zero_scalar, top_wall_dofs, V.sub(1))

# bcu_top_wall = dirichletbc(u_lid, top_wall_dofs, V)

# bcu = [bcu_left_wall, bcu_right_wall, bcu_bottom_wall, bcu_top_wall]
bcu = [bcu_inflow, bcu_bottom_wall, bcu_top_wall]

if pv_panel_flag:
    boundary_facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, internal_boundaries
    )
    boundary_dofs = locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bcu_internal_walls = dirichletbc(u_noslip, boundary_dofs, V)

    bcu.append(bcu_internal_walls)

set_bc(u_n.vector,bcu)

# Pressure Boundary Conditions
right_wall_entities = locate_entities_boundary(mesh, mesh.geometry.dim-1, right_wall)
right_wall_dofs = locate_dofs_topological(Q, mesh.geometry.dim-1, right_wall_entities)
zero_scalar = Constant(mesh, PETSc.ScalarType(0.0))
bcp_outlet = dirichletbc(zero_scalar, right_wall_dofs, Q)
bcp = [bcp_outlet]

# left_wall_dofs = locate_dofs_geometrical(Q, left_wall)
# bc_inflow = dirichletbc(PETSc.ScalarType(8), left_wall_dofs, Q)

# pin pressure with bc at corner
# upper_right_corner_pressure_dofs = locate_dofs_geometrical(Q, upper_right_corner)
# bcp_upper_right_corner = dirichletbc(PETSc.ScalarType(0.0), upper_right_corner_pressure_dofs, Q)
# bcp = [bcp_upper_right_corner]

# right_wall_dofs = locate_dofs_geometrical(Q, right_wall)
# bc_outflow = dirichletbc(PETSc.ScalarType(0), right_wall_dofs, Q)
# bcp = [bc_inflow, bc_outflow]

set_bc(p_n.vector,bcp)

# Temperature Boundary Conditions

# non-dimensional temperature
# if T0_bottom_wall != T0_top_wall:
#     DeltaT = (
#         T0_bottom_wall - T0_top_wall
#     )  # ? should this be defined as Constant(mesh, PETSc.ScalarType(bottom-top)) ?
# else:
#     DeltaT = 1 # to avoid divide by zero errors; tested on neutral state and temps stay at zero through entire sim of 0.4s
# # warning: the above logic produces different theta for the pv in neutral vs unstable ABL conditions; fix later

# reference temperature from Oeurtatani et al. 2008
# T_r = 0.5*(T0_bottom_wall + T0_top_wall) 
# T_r.x.array[:] = PETSc.ScalarType(300.0)
# T_r = Constant(mesh, PETSc.ScalarType(300.0))
T_r = Constant(mesh, PETSc.ScalarType(T_f))

# non-dim temperatures at top and bottom walls
# theta0_bottom_wall = (T0_bottom_wall - T_r) / DeltaT # from Oeurtatani et al. 2008
# theta0_top_wall = (T0_top_wall - T_r) / DeltaT # from Oeurtatani et al. 2008

# Interpolate initial temperature vertically for a smooth gradient
# T_n.interpolate(lambda x: (T0_bottom_wall + (x[1] / y_max) * (T0_top_wall - T0_bottom_wall)))
T_n.x.array[:] = PETSc.ScalarType(T_f)

# theta_n.x.array[:] = PETSc.ScalarType(T_f)

# u_.x.array[:] = PETSc.ScalarType(1.0)

# Set initial velocity?
# u_n.x.array[:] = PETSc.ScalarType(0.0)
# u_n.x[1].array[:] = PETSc.ScalarType(1.0)

# non-dimensional temperature
# theta_n.x.array[:] = (T_n.x.array[:] - T_r) / DeltaT # from Oeurtatani et al. 2008

# nonuniform temperature bc along bottom wall
# heated_cells = locate_entities(mesh, mesh.geometry.dim, lambda x: x[0] < (0.75*x_max))
rampdown_cells = locate_entities(mesh, mesh.geometry.dim, lambda x: x[0] > (0.75*x_max))
T_bottom = Function(S)
# T_bottom.interpolate(lambda x: np.full((1, x.shape[1]), T_ambient, dtype=PETSc.ScalarType)) # how do I initialize as a constant?
T_bottom.interpolate(lambda x: np.full((1, x.shape[1]), T0_bottom_wall, dtype=PETSc.ScalarType)) # how do I initialize as a constant?
bottom_wall_dofs = locate_dofs_geometrical(S, bottom_wall)
bottom_wall_temperature = LowerWallTemperature()
# T_bottom.interpolate(bottom_wall_temperature, heated_cells)
T_bottom.interpolate(bottom_wall_temperature, rampdown_cells)
bcT_bottom_wall = dirichletbc(T_bottom, bottom_wall_dofs)

# print("applying top wall temp = {}".format(T0_top_wall))
# top_wall_dofs = locate_dofs_geometrical(S, top_wall)
# bcT_top_wall = dirichletbc(
#     PETSc.ScalarType(T0_top_wall), top_wall_dofs, S
# )

left_wall_dofs = locate_dofs_geometrical(S, left_wall)
bcT_left_wall = dirichletbc(PETSc.ScalarType(T_ambient), left_wall_dofs, S)

# bcT = [bcT_top_wall, bcT_bottom_wall]
bcT = [bcT_left_wall, bcT_bottom_wall]


# exit()

# bcT = [bcT_bottom_wall, bcT_right_wall]

if pv_panel_flag:

    # theta0_panel = (T0_pv_panel - T_r) / DeltaT # from Oeurtatani et al. 2008

    print("applying pv panel temp = {}".format(T0_pv_panel))
    boundary_facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, internal_boundaries
    )
    boundary_dofs = locate_dofs_topological(S, mesh.topology.dim - 1, boundary_facets)
    bcT_internal_walls = dirichletbc(
        PETSc.ScalarType(T0_pv_panel), boundary_dofs, S
    )

    # bcT = [bcT_top_wall, bcT_bottom_wall, bcT_internal_walls]
    bcT.append(bcT_internal_walls)

set_bc(T_n.vector,bcT)
# print('bcT = ',bcT)

# bcT = [T_bc]

# Pressure Boundary Conditions from fenics code
# pressure_bc = 0
# bcp_bottom_wall = dirichletbc(PETSc.ScalarType(pressure_bc), bottom_wall_dofs, Q)
# bcp_top_wall = dirichletbc(PETSc.ScalarType(pressure_bc), top_wall_dofs, Q)
# bcp_left_wall = dirichletbc(PETSc.ScalarType(pressure_bc), left_wall_dofs, Q)
# bcp_right_wall = dirichletbc(PETSc.ScalarType(pressure_bc), right_wall_dofs, Q)

# bcp = []  # [bcp_left_wall, bcp_right_wall, bcp_bottom_wall, bcp_top_wall]
# TODO - pin pressure

# ================================================================
# Build All Forms
# ==================================================================

# step 1: tentative velocity
# chorin (removed the pressure term)
# F1 = (1 / Pr) * (
#     (1 / dt) * inner(u - u_n, v) * dx + inner(nabla_grad(u_n) * u_n, v) * dx
# )  # this might be dot not * ?
# F1 += nu * inner(nabla_grad(u), nabla_grad(v)) * dx
# F1 -= Ra * inner(theta_n * g, v) * dx

# using nu
# # # F1 = (1 / dt) * inner(u - u_n, v) * dx + inner(nabla_grad(u_n) * u_n, v) * dx
# # # F1 += nu * inner(nabla_grad(u), nabla_grad(v)) * dx
# # # F1 += beta * inner((T_n-T_r) * g, v) * dx
# F1 = (1 / dt) * inner(u - u_n, v) * dx + inner(dot(u_n, nabla_grad(u_n)), v) * dx
# F1 -= nu * inner(nabla_grad(u), nabla_grad(v)) * dx
# F1 -= beta * inner((T_n-T_r) * g, v) * dx

# using rho and mu
# F1 = (rho / dt) * inner(u - u_n, v) * dx + rho * inner(dot(u_n, nabla_grad(u_n)), v) * dx # convection
# # F1 = (rho / dt) * inner(u - u_n, v) * dx + rho * inner(nabla_grad(u_n) * u_n, v) * dx # convection
# F1 = (rho / dt) * inner(u - u_n, v) * dx + rho * (inner(nabla_grad(u_n) * u_n, v) * dx ) # convection

# Crank-Nicolson velocity
U_CN = 0.5 * (u + u_)

# Adams-Bashforth velocity
U_AB = 1.5 * u_ - 0.5 * u_n

use_pressure_in_F1 = True

F1 = (rho / dt) * inner(u - u_n, v) * dx
F1 += rho * inner(dot(U_AB, nabla_grad(U_CN)), v) * dx # convection
# # F1 += mu * inner(div(grad(u)), (v)) * dx
F1 += mu * inner(grad(U_CN), grad(v)) * dx # viscosity # + or - ??
F1 -= beta * inner((T_n-T_r) * g, v) * dx # buoyancy
if use_pressure_in_F1:
    F1 += inner(grad(p_), v) * dx

a1 = form(lhs(F1))  # dependent on u
L1 = form(rhs(F1))

# step 2: pressure correction
a2 = form(inner(nabla_grad(p), nabla_grad(q)) * dx)
if use_pressure_in_F1:
    L2 = form(dot(nabla_grad(p_), nabla_grad(q))*dx - (rho / dt) * div(u_) * q * dx)  # needs to be reassembled
else:
    L2 = form( - (rho / dt) * div(u_) * q * dx)  # needs to be reassembled
# L2 = form(-(1 / dt) * div(u_) * q * dx)  # needs to be reassembled

# step 3: velocity update
a3 = form(inner(u, v) * dx)  # doesn't need to be reassembled
if use_pressure_in_F1:
    L3 = form(inner(u_, v) * dx - (dt/rho) * inner(grad(p_ - p_n), v) * dx) # u_ is known
else:
    L3 = form(inner(u_, v) * dx - (dt/rho) * inner(grad(p_), v) * dx) # u_ is known
# L3 = form(inner(u_, v) * dx - dt * inner(nabla_grad(p_), v) * dx) # u_ is known


# # step 4: temperature
# a4 = form(
#     (1 / dt) * inner(theta, s) * dx # is theta relative to some reference temperature? when this term is removed, things get weird
#     + alpha * inner(nabla_grad(theta), nabla_grad(s)) * dx # diffusivity
#     + inner(dot(u_, nabla_grad(theta)), s) * dx # advection of temperature
#     + stab
# )  # needs to be reassembled bc of u_
# L4 = form((1 / dt) * inner(T_n, s) * dx)  # needs to be reassembled bc of T_n



# rho cp and k - wait I don't know about the dts in here
# a4 = form(
#     (rho*cp / dt) * inner(theta, s) * dx # is theta relative to some reference temperature? when this term is removed, things get weird
#     + alpha * inner(nabla_grad(theta), nabla_grad(s)) * dx # diffusivity
#     + rho*cp * inner(dot(u_, nabla_grad(theta)), s) * dx # advection of temperature
# )  # needs to be reassembled bc of u_
# L4 = form((rho*cp / dt) * inner(T_n, s) * dx)  # needs to be reassembled bc of T_n



# if stabilizing:
#     # Pe = Constant(mesh, PETSc.ScalarType(1e10))
#     h = CellDiameter(mesh)
#     unorm = sqrt(inner(u_,u_)) # ??
#     Pe = (unorm*h)/(2.0*alpha)
#     print('Peclet number = ', Pe)
#     # nb = sqrt(inner(u_,u_))
#     tau = 0.5*h*pow(4.0/(Pe*h)+2.0*unorm,-1.0)
#     s=s+tau*inner(u_,grad(s))

if stabilizing:
    # Residual, think this is just writing the "strong" governing equation?
    # T_mid = 0.5*(T_n + theta) # Crank-Nicholsen for temperature??
    r = (1 / dt)*(theta - T_n) + dot(u_, nabla_grad(theta)) - alpha*div(grad(theta))
    # r = (1 / dt)*(theta - T_n) - alpha*div(grad(theta))
    # r = dot(u_, nabla_grad(theta)) - alpha*div(grad(theta))
    # r = -alpha*div(grad(theta))
    # r = 1.0

# how to print these terms?
F4 = (1 / dt) * inner(theta - T_n, s) * dx # theta = unknown, T_n = temp from previous timestep
F4 += alpha * inner(nabla_grad(theta), nabla_grad(s)) * dx
F4 += inner(dot(u_, nabla_grad(theta)), s) * dx

if stabilizing:
    # Add SUPG stabilisation terms 
    # https://fenicsproject.org/qa/13458/how-implement-supg-properly-advection-dominated-equation/
    vnorm = sqrt(dot(u_, u_))
    h = CellDiameter(mesh)
    delta = h/(2.0*vnorm)
    stab = delta * dot(u_, grad(s)) * r * dx
    # stab = 0.0

    F4 += stab

a4 = form(lhs(F4))  # dependent on u
L4 = form(rhs(F4))

#rho cp k
# F4 = ((rho*cp) / dt) * inner(theta - T_n, s) * dx # theta = unknown, T_n = temp from previous timestep
# F4 -= k * inner(nabla_grad(theta), nabla_grad(s)) * dx # diffusion
# F4 += (rho*cp) * inner(dot(u_, nabla_grad(theta)), s) * dx # advection

# a4 = form(lhs(F4))  # dependent on u
# L4 = form(rhs(F4))

# (1/dt) * inner(T - T_, v)*dx = -inner(dot(u_, grad(T)), v)*dx -K*inner(grad(T), grad(v))*dx

# Solver for step 1
# solver1 = PETSc.KSP().create(mesh.comm)
# solver1.setType(PETSc.KSP.Type.GMRES)  # TODO - test solution with BCGS
# pc1 = solver1.getPC()
# pc1.setType(PETSc.PC.Type.HYPRE)
# pc1.setHYPREType("boomeramg")

# copied from Nav Stokes tutorial: https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
solver1 = PETSc.KSP().create(mesh.comm)
# solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.GMRES)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
# solver2 = PETSc.KSP().create(mesh.comm)
# solver2.setType(PETSc.KSP.Type.GMRES)  # TODO - test solution with BCGS
# pc2 = solver2.getPC()
# pc2.setType(PETSc.PC.Type.HYPRE)
# # pc2.setHYPREType("boomeramg") # TODO - test solution with this instead (for speed?)

# copied from Nav Stokes tutorial: https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
solver2 = PETSc.KSP().create(mesh.comm)
# solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.GMRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
# solver3 = PETSc.KSP().create(mesh.comm)
# solver3.setType(PETSc.KSP.Type.GMRES)
# pc3 = solver3.getPC()
# pc3.setType(PETSc.PC.Type.JACOBI)  # TODO - test solution with SOR
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setType(PETSc.KSP.Type.GMRES)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.JACOBI)

# copied from Nav Stokes tutorial: https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
# solver3 = PETSc.KSP().create(mesh.comm)
# # solver3.setOperators(A3)
# solver3.setType(PETSc.KSP.Type.CG)
# pc3 = solver3.getPC()
# pc3.setType(PETSc.PC.Type.SOR)

# Solver for step 4
# # solver4 = PETSc.KSP().create(mesh.comm)
# # solver4.setType(PETSc.KSP.Type.GMRES)
# # pc4 = solver4.getPC()
# # pc4.setType(PETSc.PC.Type.HYPRE)
# # pc4.setHYPREType("boomeramg")
# solver4 = PETSc.KSP().create(mesh.comm)
# solver4.setType(PETSc.KSP.Type.PREONLY)
# pc4 = solver4.getPC()
# pc4.setType(PETSc.PC.Type.LU)
# # pc4.setHYPREType("boomeramg")
solver4 = PETSc.KSP().create(mesh.comm)
solver4.setType(PETSc.KSP.Type.PREONLY)
pc4 = solver4.getPC()
pc4.setType(PETSc.PC.Type.LU)
# solver1 = PETSc.KSP().create(mesh.comm)
# # solver1.setOperators(A1)
# solver1.setType(PETSc.KSP.Type.BCGS)
# pc1 = solver1.getPC()
# pc1.setType(PETSc.PC.Type.JACOBI)

# solver = PETSc.KSP().create(domain.comm)
# solver.setOperators(A)
# solver.setType(PETSc.KSP.Type.PREONLY)
# solver.getPC().setType(PETSc.PC.Type.LU)


# ================================================================
# Begin Time Iteration
# ================================================================

eps = 3.0e-16
t = dt_num  # dt # 0.0
ct = 1  # 0
save_interval = 1 # 10  # 50

with io.XDMFFile(mesh.comm, save_fn+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_n, 0)
    xdmf.write_function(p_n, 0)
    xdmf.write_function(T_n, 0)
    # xdmf.write_function(theta_n, 0)

A1 = assemble_matrix(a1, bcs=bcu)
A1.assemble()
b1 = assemble_vector(L1)

A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = assemble_vector(L2)

A3 = assemble_matrix(a3)
A3.assemble()
b3 = assemble_vector(L3)

A4 = assemble_matrix(a4, bcs=bcT)
A4.assemble()
b4 = assemble_vector(L4)

print('shape of T_n = ',np.shape(T_n.x.array[:]))

while t < t_final + eps:
    # T_n.x.array[:] = DeltaT * theta_n.x.array[:] + T_r # from Ouertatani et al. 2008
    # T_n.interpolate(lambda x: (T0_bottom_wall + (x[1] / y_max) * (T0_top_wall - T0_bottom_wall)))
 
    # alpha.value = alpha.value*0.9
    # if alpha.value < 1e-7:
    #     alpha.value = 1e-7
    # print('alpha = ',alpha.value)

    # ================================================================
    # Assemble and Build Solvers
    # ================================================================

    A1.zeroEntries()  # resets the matrix
    A1 = assemble_matrix(A1, a1, bcs=bcu)
    A1.assemble()
    solver1.setOperators(A1)

    solver2.setOperators(A2)
    solver3.setOperators(A3)

    A4.zeroEntries()
    A4 = assemble_matrix(A4, a4, bcs=bcT)
    A4.assemble()
    solver4.setOperators(A4)

    # Step 1: Tentative velocity solve
    with b1.localForm() as loc_1:
        loc_1.set(0)
    b1 = assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.vector)
    u_.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc_2:
        loc_2.set(0)
    b2 = assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc_3:
        loc_3.set(0)
    b3 = assemble_vector(b3, L3)
    # apply_lifting(b3, [a3], [bcu])
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()
    # print('T_.x.array[0:05] = ',T_.x.array[0:5]) # how to print theta?
    # print('T_n.x.array[0:5] = ',T_n.x.array[0:5])
    # print('u_.x.array[0:5] = ',u_.x.array[0:5])
    
    # Step 4: Temperature corrrection step
    with b4.localForm() as loc_4:
        loc_4.set(0)
    b4 = assemble_vector(b4, L4)
    apply_lifting(b4, [a4], [bcT])
    b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b4, bcT)
    solver4.solve(b4, T_.vector)
    T_.x.scatter_forward()

    # Update variable with solution from this time step
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]
    T_n.x.array[:] = T_.x.array[:]
    # theta_n.x.array[:] = theta.x.array[:]

    # print(T_n.x.array[:])
    u_n_max = mesh.comm.allreduce(np.amax(u_n.vector.array), op=MPI.MAX)
    p_n_max = mesh.comm.allreduce(np.amax(p_n.vector.array), op=MPI.MAX)
    T_n_max = mesh.comm.allreduce(np.amax(T_n.vector.array), op=MPI.MAX)
    T_n_sum = mesh.comm.allreduce(np.sum(T_n.vector.array), op=MPI.SUM)

    if ct % save_interval == 0:
        with io.XDMFFile(mesh.comm, save_fn+".xdmf", "a") as xdmf:
            xdmf.write_function(u_n, t)
            xdmf.write_function(p_n, t)
            xdmf.write_function(T_n, t)
            # xdmf.write_function(theta_n, t)

        if mesh.comm.Get_rank() == 0:
            print(
                "Time = %.6f, u_max = %.6e, p_max = %.6e, T_max = %.6e, T_sum = %.6e"
                % (t, u_n_max, p_n_max, T_n_max, T_n_sum)
            )

    # Move to next step
    t += float(dt)
    ct += 1

print('alpha = ', alpha_f)
print('mu = ', mu_f)
# print('Ra = {:.2E}'.format(Ra))

# visualizing variables
# ================================================================

# print(u_n.vector.array[:])

# plt.scatter(mesh.geometry.x[:,0], mesh.geometry.x[:,1], s=10, c=p_k.vector.array[:])
# plt.scatter(coords[0, :], coords[:,1], s=10, c=p_k.vector.array[:])
# plt.show()

# coords_better = V.tabulate_dof_coordinates()
# print('size1 = ',np.shape(coords_better[0, :]))
# print('size2 = ',np.shape(coords_better[0, :]))
# plt.scatter(coords_better[:, 0], coords_better[:, 1], c=np.sqrt(u_k.vector.array[0::2]**2 + u_k.vector.array[1::2]**2))
# plt.show()