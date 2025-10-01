import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from powder.powder_stream import YuzeHuangPowderStream





@dataclass
class MeltPoolDimensions:
    """
    Stores and formats melt pool dimensions for visualization.
    """
    width: float  # meters
    length: float  # meters
    depth: float  # meters

    def get_dimension(self, axis: str) -> float:
        """Maps axis label to corresponding dimension"""
        mapping = {'X': self.width, 'Y': self.length, 'Z': self.depth}
        return mapping.get(axis.upper(), 0.0)

    @property
    def label_mapping(self) -> Dict[str, str]:
        """Maps axis labels to dimension labels"""
        return {'X': 'W', 'Y': 'L', 'Z': 'D'}


class ThermalPlotter:
    """
    Handles the visualization of thermal distributions across different planes.
    Provides consistent styling and formatting for thermal analysis plots.

    Args:
        melting_temp (float): Melting temperature of the material in Kelvin
        cmap (str, optional): Matplotlib colormap name for temperature visualization. Defaults to 'viridis'
        num_contours (int, optional): Number of contour levels for temperature visualization. Defaults to 40

    Example:
        >>> # Create plotter instance
        >>> plotter = ThermalPlotter(melting_temp=1700)  # Kelvin
        >>>
        >>> # Create and configure plot
        >>> fig, ax = plt.subplots(figsize=(10, 8))
        >>>
        >>> # Optional: Create dimensions for annotation
        >>> dimensions = MeltPoolDimensions(
        ...     width=0.001,   # 1mm
        ...     length_between=0.002,  # 2mm
        ...     depth=0.0005   # 0.5mm
        ... )
        >>>
        >>> # Generate visualization
        >>> colorbar = plotter.plot_thermal_plane(
        ...     ax=ax,
        ...     grid1=x_grid,
        ...     grid2=y_grid,
        ...     temperatures=temps,
        ...     title='XY',
        ...     xlabel='X',
        ...     ylabel='Y',
        ...     time=current_time,
        ...     speed=scan_speed,
        ...     dimensions=dimensions,          # Optional
        ...     temperature_range=(300, 2500),  # Optional
        ...     show_melt_pool=True,           # Optional
        ...     show_zero_kelvin=True,         # Optional
        ...     show_heat_source=True,         # Optional
        ...     show_grid=True,                # Optional
        ...     legend_loc='lower right'       # Optional
        ... )
        >>> plt.show()
    """

    def __init__(
            self,
            melting_temp: float,
            cmap: str = 'viridis',
            num_contours: int = 40
    ):
        """
        Initialize the ThermalPlotter with material and visualization parameters.

        Args:
            melting_temp (float): Melting temperature of the material in Kelvin
            cmap (str, optional): Matplotlib colormap for temperature visualization. Defaults to 'viridis'
            num_contours (int, optional): Number of contour levels. Defaults to 40
        """
        self.melting_temp = melting_temp
        self.cmap = cmap
        self.num_contours = num_contours

    def create_temperature_pixels(
            self,
            ax: plt.Axes,
            grid1: np.ndarray,
            grid2: np.ndarray,
            temperatures: np.ndarray,
            temperature_range: Optional[Tuple[float, float]] = None,
            label: str = 'Temperature (K)',
            interpolation: str = 'nearest'
    ) -> plt.colorbar:
        """
        Creates a pixel-based visualization of temperature distribution and adds a colorbar.
        Each data point is represented as a discrete pixel, useful for displaying raw sensor
        data or simulation results without interpolation.

        Args:
            ax (plt.Axes): Matplotlib axes object for plotting
            grid1 (np.ndarray): First coordinate meshgrid (typically X or Y)
            grid2 (np.ndarray): Second coordinate meshgrid (typically Y or Z)
            temperatures (np.ndarray): 2D array of temperature values corresponding to the grid points
            temperature_range (Tuple[float, float], optional): Min and max temperatures for color scaling.
                If None, uses the min and max of the temperature data. Defaults to None.
            label (str, optional): Label for the colorbar. Defaults to 'Temperature (K)'.
            interpolation (str, optional): Interpolation method for imshow. Common values:
                'nearest' (default): No interpolation, shows exact pixels
                'bilinear': Smooth interpolation between pixels
                'none': Matplotlib's no-interpolation mode

        Returns:
            plt.colorbar: Matplotlib colorbar object for the temperature scale

        Note:
            - The colormap is controlled by self.cmap
            - Aspect ratio is computed to maintain physical dimensions
            - Origin is set to 'lower' to match contour plot orientation
            - For sensor data visualization, 'nearest' interpolation is recommended
            - For smooth visualization, 'bilinear' interpolation might be preferred
        """
        # Calculate pixel boundary from grid points
        x_edges = np.linspace(grid1.min(), grid1.max(), temperatures.shape[1] + 1)
        y_edges = np.linspace(grid2.min(), grid2.max(), temperatures.shape[0] + 1)

        # Calculate pixel width and height for proper aspect ratio
        dx = (x_edges[-1] - x_edges[0]) / temperatures.shape[1]
        dy = (y_edges[-1] - y_edges[0]) / temperatures.shape[0]
        aspect = dx / dy

        # Set temperature limits
        if temperature_range:
            vmin, vmax = temperature_range
        else:
            vmin, vmax = np.min(temperatures), np.max(temperatures)

        # Create pixel-based temperature plot
        img = ax.imshow(
            temperatures,
            extent=[grid1.min(), grid1.max(), grid2.min(), grid2.max()],
            origin='lower',
            aspect=aspect,
            interpolation=interpolation,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax
        )

        return plt.colorbar(img, ax=ax, label=label)

    def create_temperature_contours(
        self,
        ax: plt.Axes,
        grid1: np.ndarray,
        grid2: np.ndarray,
        temperatures: np.ndarray,
        temperature_range: Optional[Tuple[float, float]] = None,
        label: str = 'Temperature (K)'
    ) -> plt.colorbar:
        """
        Creates a filled contour plot of temperature distribution and adds a colorbar.
        Automatically handles unit conversion from meters to millimeters for visualization.

        Args:
            ax (plt.Axes): Matplotlib axes object for plotting
            grid1 (np.ndarray): First coordinate meshgrid (typically X or Y)
            grid2 (np.ndarray): Second coordinate meshgrid (typically Y or Z)
            temperatures (np.ndarray): 2D array of temperature values corresponding to the grid points
            temperature_range (Tuple[float, float], optional): Min and max temperatures for color scaling.
                If None, uses the min and max of the temperature data. Defaults to None.
            label (str, optional): Label for the colorbar. Defaults to 'Temperature (K)'.

        Returns:
            plt.colorbar: Matplotlib colorbar object for the temperature scale

        Note:
            The number of contour levels is controlled by self.num_contours
            The colormap is controlled by self.cmap
        """

        # Determine contour levels based on temperature range
        if temperature_range:
            levels = np.linspace(temperature_range[0], temperature_range[1], self.num_contours)
        else:
            levels = np.linspace(np.min(temperatures), np.max(temperatures), self.num_contours)

        # Create main temperature contour plot
        contours = ax.contourf(
            grid1, grid2, temperatures,
            levels=levels, cmap=self.cmap
        )
        return plt.colorbar(contours, ax=ax, label=label)

    def add_melt_pool_boundary(
            self,
            ax: plt.Axes,
            grid1: np.ndarray,
            grid2: np.ndarray,
            temperatures: np.ndarray,
            label: str = 'Melt Pool Boundary',
    ) -> None:
        """
        Visualizes the melt pool boundary by adding a contour line at the melting temperature.
        This helps identify the region where material state changes from solid to liquid.

        Args:
            ax (plt.Axes): Matplotlib axes object for plotting
            grid1 (np.ndarray): First coordinate meshgrid
            grid2 (np.ndarray): Second coordinate meshgrid
            temperatures (np.ndarray): 2D array of temperature values corresponding to the grid points
            label (str, optional): Label for the contour line in the legend. Defaults to 'Melt Pool Boundary'.

        Note:
            - Uses red color for clear visibility of the phase change boundary
            - The melting temperature is defined by self.melting_temp
            - Grid coordinates should be in millimeters
            - Automatically adds an entry to the plot legend
        """
        ax.contour(
            grid1, grid2, temperatures,
            levels=[self.melting_temp],
            colors='r', linewidths=1,
            label=label
        )

    def add_zero_kelvin_reference(
            self,
            ax: plt.Axes,
            grid1: np.ndarray,
            grid2: np.ndarray,
            temperatures: np.ndarray,
            label: str = '0K Reference'
    ) -> None:
        """
        Adds a contour line at 0 Kelvin to identify potential numerical
        artifacts in the temperature field.

        Args:
            ax (plt.Axes): Matplotlib axes object for plotting
            grid1 (np.ndarray): First coordinate meshgrid
            grid2 (np.ndarray): Second coordinate meshgrid
            temperatures (np.ndarray): 2D array of temperature values corresponding to the grid points
            label (str, optional): Label for the contour line in the legend. Defaults to '0K Reference'.

        Note:
            - Uses white color for visibility against most backgrounds
            - Adds entry to plot legend automatically
            - Grid coordinates should already be in millimeters
        """
        ax.contour(
            grid1, grid2, temperatures,
            levels=[0],
            colors='white',
            linewidths=1,
            label=label
        )

    def add_powder_stream_boundary(
            self,
            ax: plt.Axes,
            grid1: np.ndarray,
            grid2: np.ndarray,
            params: dict,
            label: str = 'Powder Stream Boundary',
            t: float = 0.0,
    ) -> None:
        """
        Visualizes the powder stream boundary by adding a contour line at the edge of the powder stream.
        This helps identify the region where powder particles are being delivered.

        Args:
            ax (plt.Axes): Matplotlib axes object for plotting
            grid1 (np.ndarray): First coordinate meshgrid
            grid2 (np.ndarray): Second coordinate meshgrid
            params (dict): Parameters dictionary for powder stream calculations
            label (str, optional): Label for the contour line in the legend. Defaults to 'Powder Stream Boundary'.

        Note:
            - Uses purple dashed line for clear visibility of the powder stream boundary
            - The boundary is determined using is_within_powder_stream_radius function
            - Grid coordinates should be in millimeters
            - Automatically adds an entry to the plot legend
        """
        points = np.stack([grid1, grid2, np.zeros_like(grid1)], axis=-1)
        mask = YuzeHuangPowderStream.is_within_powder_stream_radius(points, params, t=t).astype(float)

        ax.contour(
            grid1*1000, grid2*1000, mask,
            levels=[0.5],
            colors='purple', linestyles='--', linewidths=1,
            label=label
        )

    def _add_dimension_label(
            self,
            ax: plt.Axes,
            xlabel: str,
            ylabel: str,
            dimensions: MeltPoolDimensions
    ) -> None:
        """Adds formatted dimension information to the plot if dimensions are provided"""
        dim_1 = dimensions.get_dimension(xlabel.upper())
        dim_2 = dimensions.get_dimension(ylabel.upper())
        label_1 = dimensions.label_mapping[xlabel.upper()]
        label_2 = dimensions.label_mapping[ylabel.upper()]

        # Create dimension text with white background for readability
        ax.text(
            0.02, 0.98,
            f'{label_1} × {label_2}: {dim_1 * 1000:.2f} × {dim_2 * 1000:.2f} mm',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7)
        )

    def add_heat_source_marker(
            self,
            ax: plt.Axes,
            xlabel: str,
            ylabel: str,
            current_position: float,
            label: str = 'Heat Source'
    ) -> None:
        """
        Adds a marker showing the current position of the heat source on the chosen plane.
        For moving heat sources, the position is calculated based on time and speed.

        Args:
            ax (plt.Axes): Matplotlib axes object for plotting
            xlabel (str): Label of x-axis ('X', 'Y', or 'Z')
            ylabel (str): Label of y-axis ('X', 'Y', or 'Z')
            time (float): Current simulation time in seconds
            speed (float): Process speed in meters per second
            label (str, optional): Label for the marker in the legend. Defaults to 'Heat Source'.

        Note:
            - Marker is only shown on planes containing the Y-axis (movement direction)
            - Uses a red 'x' marker for visibility
            - Position is calculated as (speed * time) and converted to millimeters
            - For XY plane: marker appears on Y-axis (x=0)
            - For YZ plane: marker appears on Y-axis (z=0)
        """

        if ylabel.upper() == 'Y':
            ax.plot(0, current_position, 'rx', markersize=5, label=label)
        if xlabel.upper() == 'Y':
            ax.plot(current_position, 0, 'rx', markersize=5, label=label)

    def plot_thermal_plane(
            self,
            ax: plt.Axes,
            grid1: np.ndarray,
            grid2: np.ndarray,
            temperatures: np.ndarray,
            title: str,
            xlabel: str,
            ylabel: str,
            time: float,
            speed: float,
            dimensions: Optional[MeltPoolDimensions] = None,
            temperature_range: Optional[Tuple[float, float]] = None,
            show_melt_pool: bool = True,
            show_zero_kelvin: bool = True,
            show_heat_source: bool = True,
            show_powder_stream_boundary: bool = True,
            show_grid: bool = True,
            legend_loc: str = 'lower right',
            plot_mode='contour',
            params=None,
    ) -> plt.colorbar:
        """
        Creates a comprehensive thermal plane visualization with configurable components.
        Handles unit conversion and provides a consistent interface for all plotting features.

        Args:
            ax (plt.Axes): Matplotlib axes for plotting
            grid1 (np.ndarray): First coordinate meshgrid in meters (typically X or Y)
            grid2 (np.ndarray): Second coordinate meshgrid in meters (typically Y or Z)
            temperatures (np.ndarray): 2D array of temperature values at grid points
            title (str): Plot title, will be suffixed with 'Plane'
            xlabel (str): X-axis label, must be 'X', 'Y', or 'Z'
            ylabel (str): Y-axis label, must be 'X', 'Y', or 'Z'
            time (float): Current simulation time in seconds
            speed (float): Process speed in meters per second
            dimensions (Optional[MeltPoolDimensions]): Object containing melt pool dimensions for annotation.
                If provided, adds dimension labels to the plot. Defaults to None.
            temperature_range (Optional[Tuple[float, float]]): Min and max temperatures for color scaling.
                If None, uses the min and max of the temperature data. Defaults to None.
            show_melt_pool (bool): Whether to show the melt pool boundary contour. Defaults to True.
            show_zero_kelvin (bool): Whether to show the 0K reference contour. Defaults to True.
            show_heat_source (bool): Whether to show the heat source position marker. Defaults to True.
            show_grid (bool): Whether to show the background grid. Defaults to True.
            legend_loc (str): Location of the legend. Defaults to 'lower right'.

        Returns:
            plt.colorbar: Matplotlib colorbar object for the temperature scale

        Note:
            - All grid coordinates are converted from meters to millimeters for visualization
            - The plot automatically includes appropriate labels and units
            - The legend is only shown if at least one optional component is visible
            - Grid lines are semi-transparent for better readability
        """
        # Convert grids from meters to millimeters once for all subsequent uses

        grid1_mm = grid1 * 1000
        grid2_mm = grid2 * 1000

        # Get the appropriate plotting function or raise error for invalid mode
        try:
            colorbar = {
                'contour': self.create_temperature_contours,
                'pixel': self.create_temperature_pixels
            }[plot_mode](
                    ax, grid1_mm, grid2_mm, temperatures, temperature_range
                )
        except KeyError:
            raise ValueError(
                f"Invalid plot mode '{plot_mode}'. "
                f"Supported modes are: contour, pixel"
            )

        # Add optional visualization components
        if show_melt_pool:
            self.add_melt_pool_boundary(ax, grid1_mm, grid2_mm, temperatures)

        if show_zero_kelvin:
            self.add_zero_kelvin_reference(ax, grid1_mm, grid2_mm, temperatures)

        # Add dimension information if provided
        if dimensions is not None:
            self._add_dimension_label(ax, xlabel, ylabel, dimensions)

        # Add heat source marker if requested
        if show_heat_source:
            current_position = time * speed * 1000
            self.add_heat_source_marker(ax, xlabel, ylabel, current_position)

        if show_powder_stream_boundary:
            self.add_powder_stream_boundary(ax, grid1, grid2, params, t=time)

        # Configure plot aesthetics
        ax.set_title(f'{title} Plane')
        ax.set_xlabel(f'{xlabel} (mm)')
        ax.set_ylabel(f'{ylabel} (mm)')

        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.3)

        # Only show legend if at least one optional component is visible
        if show_melt_pool or show_zero_kelvin or show_heat_source:
            ax.legend(loc=legend_loc)

        return colorbar
