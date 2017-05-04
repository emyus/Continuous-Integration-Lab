# -*- coding: utf-8 -*-
"""
Finite Elements Coursework for Simulation and Modelling Assignment #2 2016

README:
There are two classes, FiniteElements and the assosciated FiniteElementsTest class.

The unit-testing is being run from within main() as is the code to generate
the required plots.

The code required to generate the mesh and associated functionality have been
copied from the assignment description.

@author: P.Bartram@soton.ac.uk
"""

import numpy as np
import unittest
import sympy

print('#################################################################')
print('Congratulations, <your name>, the Finite Element code is running successfully!')
print('#################################################################')

print('***checking the POLL function**')

class FiniteElements():
    ''' This class will perform finite element analysis on a specified grid. '''

    def _CalculateShapeFunctions(self, inputs):
        ''' Calculate the values of a given shape function for the xi and eta
        values specified.

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------
            - inputs : np.array(float) length 2
            The location within the given reference triangle.
        Returns
        ----------
            - N0, N1, N2
            The shape function values themselves.
        '''
        assert type(inputs) == np.ndarray or type(inputs) == list, \
            'inputs is not of type np.ndarray / list but {}'.format(type(inputs))
        assert len(inputs) == 2, \
            'length of inputs is not 2 but {}'.format(len(inputs))
        xi, eta = inputs
        N0 = 1 - eta - xi
        N1 = xi
        N2 = eta
        return N0, N1, N2

    def _CalculateShapeFunctionDerivatives(self):
        ''' Returns the derivatives of the shape functions.

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Returns
        ----------
            - tuple: de, dn - each of three elements
            The derivatives of the shape function.
        '''
        de = -1, 1, 0
        dn = -1, 0, 1
        return de, dn

    def _CalculateGlobalCoordinatesGivenLocalCoordinates(self, X, localCoordinates):
        ''' Calculate the global coordinates given the element node locations and
        the local coordinates within that element

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------
            - X : np.array(3,2)
            The global coordinates of the element nodes.
            - localCoordinates: np.array(2)
            The local coordinates specified in terms of the reference triangle.
        Returns
        ----------
            - tuple(2) : (global_x, global_y)
            The global location of the specified point on the mesh.
        '''
        assert type(X) == np.ndarray or type(X) == list, \
            'X is not of type np.ndarray / list but {}'.format(type(X))
        assert type(localCoordinates) == np.ndarray or type(localCoordinates) == list, \
            'localCoordinates is not of type np.ndarray / list but {}'.format(type(localCoordinates))
        assert len(localCoordinates) == 2, \
            'length of localCoordinates is not 2 but {}'.format(len(localCoordinates))
        assert len(X) == 3, \
            'length of X is not 3 but {}'.format(len(X))
        assert len(X[0]) == 2, \
            'length of X[0] is not 2 but {}'.format(len(X[0]))
        x = X[:,0]
        y = X[:,1]

        # Unpack all of our inputs and name properly.
        x0, x1, x2 = x
        y0, y1, y2 = y

        # Calculate the shape functions based upon the local specified coordinates.
        shapeFunctions = self._CalculateShapeFunctions(localCoordinates)

        # Extract and rename fields for clarity
        N0, N1, N2 = shapeFunctions

        # Calculate the global coordinates
        global_x = x0*N0 + x1*N1 + x2*N2
        global_y = y0*N0 + y1*N1 + y2*N2

        # Return the global coordinates.
        return global_x, global_y

    def _CalculateJacobianGivenLocalCoordinates(self, X, localCoordinates):
        ''' Calculate the Jacobian matrix at the location specified.

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------
            - X : np.array(3,2)
            The global coordinates of the element nodes.
            - localCoordinates: np.array(2)
            The local coordinates specified in terms of the reference triangle.
        Returns
        ----------
            - np.array(2,2) : J
            The Jacobian matrix at that location.
        '''
        assert type(X) == np.ndarray or type(X) == list, \
            'X is not of type np.ndarray / list but {}'.format(type(X))
        assert type(localCoordinates) == np.ndarray or type(localCoordinates) == list, \
            'localCoordinates is not of type np.ndarray / list but {}'.format(type(localCoordinates))
        assert len(localCoordinates) == 2, \
            'length of localCoordinates is not 2 but {}'.format(len(localCoordinates))
        assert len(X) == 3, \
            'length of X is not 3 but {}'.format(len(X))
        assert len(X[0]) == 2, \
            'length of X[0] is not 2 but {}'.format(len(X[0]))
        x = X[:,0]
        y = X[:,1]

        x0, x1, x2 = x
        y0, y1, y2 = y

        shapeDerivatives = self._CalculateShapeFunctionDerivatives()
        de, dn = shapeDerivatives

        dx_de = x0*de[0] + x1*de[1] + x2*de[2]
        dx_dn = x0*dn[0] + x1*dn[1] + x2*dn[2]
        dy_de = y0*de[0] + y1*de[1] + y2*de[2]
        dy_dn = y0*dn[0] + y1*dn[1] + y2*dn[2]

        J = np.array([[dx_de, dy_de],[dx_dn, dy_dn]])
        return J

    def _CalculateJacobianDeterminantGivenLocalCoordinates(self, X, localCoordinates):
        ''' Calculate the determinant of the Jacobian matrix at the location specified.

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------
            - X : np.array(3,2)
            The global coordinates of the element nodes.
            - localCoordinates: np.array(2)
            The local coordinates specified in terms of the reference triangle.
        Returns
        ----------
            - det(J)
            The determinant of the Jacobian matrix at the location specified.
        '''
        assert type(X) == np.ndarray or type(X) == list, \
            'X is not of type np.ndarray / list but {}'.format(type(X))
        assert type(localCoordinates) == np.ndarray or type(localCoordinates) == list, \
            'localCoordinates is not of type np.ndarray / list but {}'.format(type(localCoordinates))
        assert len(localCoordinates) == 2, \
            'length of localCoordinates is not 2 but {}'.format(len(localCoordinates))
        assert len(X) == 3, \
            'length of X is not 3 but {}'.format(len(X))
        assert len(X[0]) == 2, \
            'length of X[0] is not 2 but {}'.format(len(X[0]))
        J = self._CalculateJacobianGivenLocalCoordinates(X, localCoordinates)
        return np.linalg.det(J)

    def _CalculateDerivativesOfShapeFunctionsInGlobalCoordinates(self, X, localCoordinates):
        ''' Calculate the derivative of the shape function at the location specified.

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------
            - X : np.array(3,2)
            The global coordinates of the element nodes.
            - localCoordinates: np.array(2)
            The local coordinates specified in terms of the reference triangle.
        Returns
        ----------
            - det(J)
            The derivatives of the shape function.
        '''
        assert type(X) == np.ndarray or type(X) == list, \
            'X is not of type np.ndarray / list but {}'.format(type(X))
        assert type(localCoordinates) == np.ndarray or type(localCoordinates) == list, \
            'localCoordinates is not of type np.ndarray / list but {}'.format(type(localCoordinates))
        assert len(localCoordinates) == 2, \
            'length of localCoordinates is not 2 but {}'.format(len(localCoordinates))
        assert len(X) == 3, \
            'length of X is not 3 but {}'.format(len(X))
        assert len(X[0]) == 2, \
            'length of X[0] is not 2 but {}'.format(len(X[0]))
        derivatives = self._CalculateShapeFunctionDerivatives()
        de, dn = derivatives

        retVal = np.zeros([3,2])
        J = self._CalculateJacobianGivenLocalCoordinates(X, localCoordinates)

        for i in range(3):
            dN = np.array([de[i], dn[i]])

            retVal[i, :] = np.linalg.solve(J, dN)

        return retVal

    def _PerformQuadratureOnReferenceTriangle(self, psi):
        ''' Perform quadrature on our reference triangle

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------
            - psi : function
            The function to perform the quadrature over.
        Returns
        ----------
            - accum
            Integral of function over the reference triangle.
        '''
        assert callable(psi), \
            'psi is not a function but is of type {}'.format(type(psi))

        xi = np.array([1/6, 4/6, 1/6])
        eta = np.array([1/6, 1/6, 4/6])

        accum = 0
        for i in range(3):
            state = np.array([xi[i], eta[i]])
            accum += psi(state)
        accum /= 6
        return accum

    def _PerformQuadratureOnElement(self, phi, globalCoords):
        ''' Perform quadrature on a particular element.

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------
            - phi : function
            The function to perform the quadrature over.
            - globalCoords : np.array(3,2)
            The global coordinates of the element to integrate over.
        Returns
        ----------
            - Integral of function over the element
        '''
        assert callable(phi), \
            'phi is not a function but is of type {}'.format(type(phi))
        assert np.shape(globalCoords) == (3,2), \
            'globalCoords is not of shape (3,2) but {}'.format(np.shape(globalCoords))
        psi = lambda  xi_in: self._CalculateJacobianDeterminantGivenLocalCoordinates(globalCoords, xi_in)* \
                          phi(self._CalculateGlobalCoordinatesGivenLocalCoordinates(globalCoords, xi_in),
                             self._CalculateDerivativesOfShapeFunctionsInGlobalCoordinates(globalCoords, xi_in),
                             self._CalculateShapeFunctions(xi_in))

        return self._PerformQuadratureOnReferenceTriangle(psi)

    def _CalculateStiffness(self, globalCoords):
        ''' Calculate the stiffness matrix for a given element

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------
            - globalCoords : np.array(3,2)
            The global coordinates of the element to calculate for.
        Returns
        ----------
            - stiffness : np.array(3,3)
        '''
        assert np.shape(globalCoords) == (3,2), \
            'globalCoords is not of shape (3,2) but {}'.format(np.shape(globalCoords))
        stiffness = np.zeros([3,3])
        for a in range(3):
            for b in range(3):
                stiff_calc = lambda x, ds, s: ds[a,0] * ds[b,0] +  ds[a,1] * ds[b,1]
                stiffness[a,b] = self._PerformQuadratureOnElement(stiff_calc, globalCoords)
        return stiffness

    def _CalculateForce(self, globalCoords, f):
        ''' Calculate the force vector for a given element

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------
            - globalCoords : np.array(3,2)
            The global coordinates of the element to calculate for.
        Returns
        ----------
            - force : np.array(3)
        '''
        assert callable(f), \
            'f is not a function but is of type {}'.format(type(f))
        assert np.shape(globalCoords) == (3,2), \
            'globalCoords is not of shape (3,2) but {}'.format(np.shape(globalCoords))
        force = np.zeros(3)
        for b in range(3):
            forceCalc = lambda x, ds, s: s[b]*f(x)
            force[b] = self._PerformQuadratureOnElement(forceCalc, globalCoords)

        return force

    def PerformFiniteElement(self, nodes, IEN, ID, f):
        ''' This is the main interface function to perform finite element
        analysis on the grid and function provided.

        Parameters
        ----------
            nodes : array of float
                (Nnodes, 2) array containing the x, y coordinates of the nodes
            IEN : array of int
                (Nelements, 3) array linking element number to node number
            ID : array of int
                (Nnodes,) array linking node number to equation number; value is -1 if node should not appear in global
            f : function
            The function describing the heat in the system.
        Returns
        ----------
            - output : np.array(Nnodes)
            Solution to the generated system of equations.
        '''
        assert type(nodes) == np.ndarray or type(nodes) == list, \
            'nodes is not of type np.ndarray / list but {}'.format(type(nodes))
        assert type(IEN) == np.ndarray or type(IEN) == list, \
            'IEN is not of type np.ndarray / list but {}'.format(type(IEN))
        assert type(ID) == np.ndarray or type(ID) == list, \
            'ID is not of type np.ndarray / list but {}'.format(type(ID))
        assert callable(f), \
            'f is not of type function but of type {}'.format(type(f))

        # Configure storage for K and F global.
        Nelements = IEN.shape[0]
        Nequations = np.max(ID)+1
        Nnodes = nodes.shape[0]
        K = np.zeros([Nequations, Nequations])
        F = np.zeros(Nequations)

        # Location matrix
        LM = np.zeros_like(IEN.T)
        for e in range(Nelements):
            for a in range(3):
                LM[a,e] = ID[IEN[e,a]]

        for e in range(Nelements):
            # Calculate local stiffness matrix
            stiffnessLocal = self._CalculateStiffness(nodes[IEN[e,:],:])

            # Map from local stiffness to global stiffness
            for a in range(3):
                A = LM[a, e]
                for b in  range(3):
                    B = LM[b,e]
                    if A != -1 and  B != -1:
                        K[A, B] = K[A, B] + stiffnessLocal[a,b]

        for e in range(Nelements):
            # Calculate force
            forceLocal = self._CalculateForce(nodes[IEN[e,:],:], f)

            # Map from local force vector to global force vector.
            for a in range(3):
                A = LM[a, e]
                if A != -1:
                    F[A]  += forceLocal[a]

        # Solve system of equations
        T = np.linalg.solve(K,F)

        # Apply boundary conditions to nodes marked as -1.
        output = np.zeros(Nnodes)
        for n in range(Nnodes):
            if ID[n] >= 0:
                output[n] = T[ID[n]]


        return output

    def _find_node_index_of_location(self, nodes, location):
        """
        Given all the nodes and a location (that should be the location of *a* node), return the index of that node.

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------

        nodes : array of float
            (Nnodes, 2) array containing the x, y coordinates of the nodes
        location : array of float
            (2,) array containing the x, y coordinates of location
        """
        dist_to_location = np.linalg.norm(nodes - location, axis=1)
        return np.argmin(dist_to_location)


    def _generate_g_grid(self, side_length):
        """
        Generate a 2d triangulation of the letter G. All triangles have the same size (right triangles,
        short length side_length)

        -----------------------------------------------------------------------
        This is not an interface function and should only be called by
        functions inside the FiniteElements class.
        -----------------------------------------------------------------------

        Parameters
        ----------

        side_length : float
            The length of each triangle. Should be 1/N for some integer N

        Returns
        -------

        nodes : array of float
            (Nnodes, 2) array containing the x, y coordinates of the nodes
        IEN : array of int
            (Nelements, 3) array linking element number to node number
        ID : array of int
            (Nnodes,) array linking node number to equation number; value is -1 if node should not appear in global arrays.
        """
        x = np.arange(0, 4+0.5*side_length, side_length)
        y = np.arange(0, 5+0.5*side_length, side_length)
        X, Y = np.meshgrid(x,y)
        potential_nodes = np.zeros((X.size,2))
        potential_nodes[:,0] = X.ravel()
        potential_nodes[:,1] = Y.ravel()
        xp = potential_nodes[:,0]
        yp = potential_nodes[:,1]
        nodes_mask = np.logical_or(np.logical_and(xp>=2,np.logical_and(yp>=2,yp<=3)),
                                      np.logical_or(np.logical_and(xp>=3,yp<=3),
                                                       np.logical_or(xp<=1,
                                                                        np.logical_or(yp<=1, yp>=4))))
        nodes = potential_nodes[nodes_mask, :]
        ID = np.zeros(len(nodes), dtype=np.int)
        n_eq = 0
        for nID in range(len(nodes)):
            if np.allclose(nodes[nID,0], 4):
                ID[nID] = -1
            else:
                ID[nID] = n_eq
                n_eq += 1
        inv_side_length = int(1 / side_length)
        Nelements_per_block = inv_side_length**2
        Nelements = 2 * 14 * Nelements_per_block
        IEN = np.zeros((Nelements,3), dtype=np.int)
        block_corners = [[0,0], [1,0], [2,0], [3,0],
                         [0,1],               [3,1],
                         [0,2],        [2,2], [3,2],
                         [0,3],
                         [0,4], [1,4], [2,4], [3,4]]
        current_element = 0
        for block in block_corners:
            for i in range(inv_side_length):
                for j in range(inv_side_length):
                    node_locations = np.zeros((4,2))
                    for a in range(2):
                        for b in range(2):
                            node_locations[a+2*b,0] = block[0] + (i+a)*side_length
                            node_locations[a+2*b,1] = block[1] + (j+b)*side_length
                    index_lo_l = self._find_node_index_of_location(nodes, node_locations[0,:])
                    index_lo_r = self._find_node_index_of_location(nodes, node_locations[1,:])
                    index_hi_l = self._find_node_index_of_location(nodes, node_locations[2,:])
                    index_hi_r = self._find_node_index_of_location(nodes, node_locations[3,:])
                    IEN[current_element, :] = [index_lo_l, index_lo_r, index_hi_l]
                    current_element += 1
                    IEN[current_element, :] = [index_lo_r, index_hi_r, index_hi_l]
                    current_element += 1
        return nodes, IEN, ID


