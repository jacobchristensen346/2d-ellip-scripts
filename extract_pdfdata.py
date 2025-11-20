"""
-----------
pdfellip.py
-----------

This Python module aids in data analysis of a wafer scan
on the Gaertner 2D Ellipsometer.
Coordinate and thickness data is extracted from the pdf
output, and a meshgrid is interpolated for the purpose
of contour plot data visualization.

Classes:
    EllipMap: Extract and manipulate 2D Ellipsometer data.
        Prepares data for contour plot visualization.
"""

# This script extracts data from the pdf 
# as saved on the Gaertner 2D Ellipsometer
# for a particular scan/run

# imports
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np
import PyPDF2
import re


class EllipMap:
    """Extracts relevant statistical information from
    the pdf output on the Gaertner 2D Ellipsometer.

    Includes an interpolation function for creating either
    a cartesian or polar meshgrid appropriate for contour plots.

    Attributes:
        stddev (float): The standard deviation of the
            thickness read by the ellipsometer.
        mean (float): The average thickness read
            by the ellipsometer.
        rad_arr (np.ndarray): Array that contains the
            extracted radial coordinates for a
            particular scan/run.
        theta_arr (np.ndarray): Array that contains the
            extracted theta (angular) coordinates for
            a particular scan/run.
        thick_arr (np.ndarray): Array that contains the
            extracted thickness measurements at each
            radial and angular coordinate.
    """

    def __init__(self, pdfpath):
        """Read in pdf file and extract relevant information.

        Information extracted includes the mean, standard deviation,
        radial coordinates, angular coordinates, and
        thickness measurements for each coordinate pair.

        Args:
            pdfpath (str): The directory filepath to the
                pdf of interest.
                
        Returns:
            None.
        """
        all_content = ''  # string to append statistics to
        # Open the pdf file using PyPDF2.
        # Iterate through each page, extract
        # the data as text, and append to a string.
        with open(pdfpath, "rb") as pdf_file:
            read_pdf = PyPDF2.PdfReader(pdf_file)
            num_pages = len(read_pdf.pages)
            for page_num in np.arange(num_pages):
                page = read_pdf.pages[int(page_num)]
                page_content = page.extract_text()
                all_content += page_content

        # Define the regex used to filter the
        # string of all data.
        patt_stddev = r"StdDev\s*(.*?)\s" 
        patt_mean = r"Mean\s*(.*?)\s" 
        patt_rad = r"R=\s*(.*?),"
        patt_theta = r"Theta=\s*(.*?),"
        # Include the possibility of 'No Solution' for thickness.
        patt_thick = r"Thick1=\s*(.*?),|No Soution"

        # Find the mean and standard deviation
        # using the re.search function.
        self.stddev = float(re.search(patt_stddev, all_content).group(1))
        self.mean = float(re.search(patt_mean, all_content).group(1))

        # Now for radial, angular, and thickness data
        # find all the matches from each regex input
        # and place the matches into a list.
        matches_rad = re.findall(patt_rad, all_content)
        matches_theta = re.findall(patt_theta, all_content)
        # Replace instances of 'No Solution' with nan.
        # 'No Solution' is returned as '' from our regex algorithm.
        matches_thick = np.array(re.findall(patt_thick, all_content))
        matches_thick[matches_thick == ''] = np.nan

        # transform list into np.ndarray with floats
        self.rad_arr = np.array(matches_rad).astype(float)
        self.theta_arr = np.radians(np.array(matches_theta).astype(float))

        # We have a bit more work to do for thicknesses.
        # After transforming to floats, identify any nan values
        # and replace them with linearly interpolated values.
        self.thick_arr = np.array(matches_thick).astype(float, casting='unsafe')
        mask = np.isnan(self.thick_arr)
        self.thick_arr[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            self.thick_arr[~mask])

    def interp_grid(self, polar_coords, ptnum):
        """Create a meshgrid and interpolate.

        The Radial Basis Function is used to interpolate
        over a fine meshgrid in preparation for data
        manipulation and contour plotting.

        Args:
            polar_coords (bool): If True, the meshgrid
                will be returned as polar coordinates.
                If False, coordinates remain cartesian.
            ptnum (int): The number of points along
                either axis in the meshgrid, resulting
                in a grid of (ptnum * ptnum) total points.

        Returns:
            grid1 (np.ndarray): 2D array of either the
                x-cartesian or radial coordinates of the meshgrid.
            grid2 (np.ndarray): 2D array of either the
                y-cartesian or theta coordinates of the meshgrid.
            z (np.ndarray): The z-values associated with
                each coordinate pairing in the meshgrid.
        """
        # Convert r and theta to cartesian (x and y).
        # This is necessary for proper interpolation behavior.
        x = self.rad_arr * np.cos(self.theta_arr)
        y = self.rad_arr * np.sin(self.theta_arr)
        # Pair each x and y point in a new array.
        pairings = np.stack([x, y], axis=-1)
        
        # Create the Radial Basis Function interpolation.
        # This interpolation places a kernel function at each known point
        # and solves for weights such that each point's value is satisfied.
        # The resulting functions and weights are used for interpolation.
        rbf = scipy.interpolate.RBFInterpolator(pairings, self.thick_arr,
                                                kernel='cubic')
        
        # Create a denser meshgrid to plot on.
        x_dense = np.linspace(min(x), max(x), ptnum)
        y_dense = np.linspace(min(y), max(y), ptnum)
        x_grid, y_grid = np.meshgrid(x_dense, y_dense)
        # Flatten the 2D grid arrays and pair the coordinates.
        # This is needed since RBFInterpolate expects a single 2D array.
        grid_pairs = np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)
        
        # Apply the interpolation function to the paired grid points.
        z_interp_flat = rbf(grid_pairs)
        # Reshape the pairings to the original grid shape.
        z = z_interp_flat.reshape(x_grid.shape)

        if polar_coords == True:
            # Convert the x and y grid to a polar grid.
            r_grid = np.sqrt(x_grid**2 + y_grid**2)
            theta_grid = np.arctan2(y_grid, x_grid)
            grid1, grid2 = r_grid, theta_grid
        else:
            # Keep meshgrid in cartesian.
            grid1, grid2 = x_grid, y_grid
            
        return grid1, grid2, z
