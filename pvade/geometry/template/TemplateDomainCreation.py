import gmsh
import numpy as np


class TemplateDomainCreation:
    """This class creates the geometry used for a given example.
    Gmsh is used to create the computational domain

    """

    def __init__(self, params):
        """The class is initialised here

        Args:
            params (_type_): _description_
        """

        # Get MPI communicators
        self.comm = params.comm
        self.rank = params.rank
        self.num_procs = params.num_procs

        # Initialize Gmsh options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        # All ranks create a Gmsh model object
        self.gmsh_model = gmsh.model()
        self.gmsh_model.add("domain")
        self.gmsh_model.setCurrent("domain")

    def build(self, params):
        """
            panels: This function creates the computational domain for a 3d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.
            panels2d: This function creates the computational domain for a 2d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.
            cylinder3d: This function creates the computational domain for a flow around a 3D cylinder.
            cylinder2d: This function creates the computational domain for a flow around a 2D cylinder.
        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """
        pass

    def set_length_scales(self, params, domain_markers):
        """This function call defines the characteristic length for the mesh in locations of interst
        LcMin,LcMax,DistMin and DistMax are used to create a refined mesh in specific locations
        which results in a high fidelity mesh without using a uniform element size in the whole mesh.
        """
        if self.rank == 0:
            all_pts = self.gmsh_model.occ.getEntities(0)
            self.gmsh_model.mesh.setSize(all_pts, params.domain.l_char)

    def mark_surfaces(self, params, domain_markers):
        """This function call iterates over all boundaries and assigns tags for each boundary.
        The Tags are being used when appying boundary condition.
        """

        self.ndim = self.gmsh_model.get_dimension()

        # Surfaces are the entities with dimension 1 less than the mesh dimension
        # i.e., surfaces have dim=2 (facets) on a 3d mesh
        # and dim=1 (lines) on a 2d mesh
        surf_ids = self.gmsh_model.occ.getEntities(self.ndim - 1)

        for surf in surf_ids:
            tag = surf[1]

            com = self.gmsh_model.occ.getCenterOfMass(self.ndim - 1, tag)

            if np.isclose(com[0], params.domain.x_min):
                domain_markers["x_min"]["gmsh_tags"].append(tag)

            elif np.allclose(com[0], params.domain.x_max):
                domain_markers["x_max"]["gmsh_tags"].append(tag)

            elif np.allclose(com[1], params.domain.y_min):
                domain_markers["y_min"]["gmsh_tags"].append(tag)

            elif np.allclose(com[1], params.domain.y_max):
                domain_markers["y_max"]["gmsh_tags"].append(tag)

            elif self.ndim == 3 and np.allclose(com[2], params.domain.z_min):
                domain_markers["z_min"]["gmsh_tags"].append(tag)

            elif self.ndim == 3 and np.allclose(com[2], params.domain.z_max):
                domain_markers["z_max"]["gmsh_tags"].append(tag)

            else:
                domain_markers["internal_surface"]["gmsh_tags"].append(tag)


        # self.gmsh_model.addPhysicalGroup(self.ndim, [1], domain_marker_idx["fluid"])
        # self.gmsh_model.setPhysicalName(self.ndim, domain_marker_idx["fluid"], "fluid")
        # TODO: this is a hack to add fluid tags, need to loop through cells 
        # as we do for facets and mark fluid and structure 
        domain_markers["fluid"]["gmsh_tags"].append(1)

        for key, data in domain_markers.items():
            if len(data["gmsh_tags"]) > 0:
                # Cells (i.e., entities of dim = msh.topology.dim)
                if data["entity"] == "cell":
                    self.gmsh_model.addPhysicalGroup(self.ndim, data["gmsh_tags"], data["idx"])
                    self.gmsh_model.setPhysicalName(self.ndim, data["idx"], key)

                # Facets (i.e., entities of dim = msh.topology.dim - 1)
                if data["entity"] == "facet":
                    self.gmsh_model.addPhysicalGroup(self.ndim-1, data["gmsh_tags"], data["idx"])
                    self.gmsh_model.setPhysicalName(self.ndim-1, data["idx"], key)

        return domain_markers

